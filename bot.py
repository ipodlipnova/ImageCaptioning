import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aiogram import Bot, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types.chat import ChatActions
from aiogram.types.message import ContentType
from aiogram.utils import executor

import image_captioning_model
from config import TOKEN


class CaptionNet(nn.Module):
    def __init__(self, n_tokens, emb_size=128, lstm_units=256, cnn_feature_size=2048):
        super(self.__class__, self).__init__()

        # стандартная архитектура такой сети такая:
        # 1. линейные слои для преобразования эмбеддиинга картинки в начальные состояния h0 и c0 LSTM-ки
        # 2. слой эмбедднга
        # 3. несколько LSTM слоев (для начала не берите больше двух, чтобы долго не ждать)
        # 4. линейный слой для получения логитов
        self.cnn_to_h0 = nn.Linear(cnn_feature_size, lstm_units)
        self.cnn_to_c0 = nn.Linear(cnn_feature_size, lstm_units)

        self.emb = nn.Embedding(n_tokens, emb_size)

        self.lstm = nn.LSTM(emb_size, lstm_units, batch_first=True)
        self.logits = nn.Linear(lstm_units, n_tokens)

    def forward(self, image_vectors, captions_ix):
        """
        Apply the network in training mode.
        :param image_vectors: torch tensor, содержащий выходы inseption. Те, из которых будем генерить текст
                shape: [batch, cnn_feature_size]
        :param captions_ix:
                таргет описания картинок в виде матрицы
        :returns: логиты для сгенерированного текста описания, shape: [batch, word_i, n_tokens]

        Обратите внимание, что мы подаем сети на вход сразу все префиксы описания
        и просим ее к каждому префиксу сгенерировать следующее слово!
        """

        # 1. инициализируем LSTM state
        # 2. применим слой эмбеддингов к image_vectors
        # 3. скормим LSTM captions_emb
        # 4. посчитаем логиты из выхода LSTM
        initial_cell = self.cnn_to_c0(image_vectors)
        initial_hid = self.cnn_to_h0(image_vectors)

        captions_emb = self.emb(captions_ix)

        batch_size, caption_len, emb_size = captions_emb.size()
        caption_len = captions_emb.size()[1]
        len_list = [caption_len for i in range(captions_emb.size()[0])]
        input_seq = nn.utils.rnn.pack_padded_sequence(captions_emb, len_list, batch_first=True)

        lstm_out, _ = self.lstm(input_seq, (initial_cell[None], initial_hid[None]))
        lstm_out = lstm_out.data.view(caption_len, batch_size, -1).permute(1, 0, 2)

        logits = self.logits(lstm_out)

        return logits


class BahdanauAttention(nn.Module):
    """ Class performs Additive Bahdanau Attention.
        Source: https://arxiv.org/pdf/1409.0473.pdf

    """

    def __init__(self, num_features, hidden_dim, output_dim=1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, decoder_hidden):
        """
        Arguments:
        ----------
        - features - features returned from Encoder
        - decoder_hidden - hidden state output from Decoder

        Returns:
        ---------
        - context - context vector with a size of (1,2048)
        - atten_weight - probabilities, express the feature relevance
        """
        # add additional dimension to a hidden (required for summation)

        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combine result from 2 fc layers
        atten_tan = torch.tanh(atten_1 + atten_2)
        atten_score = self.v_a(atten_tan)
        atten_weight = F.softmax(atten_score, dim=1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        # the size of context equals to a number of feature maps
        context = torch.sum(atten_weight * features, dim=1)
        atten_weight = atten_weight.squeeze(dim=2)

        return context, atten_weight


class DecoderRNN(nn.Module):
    """Attributes:
         - embedding_dim - specified size of embeddings;
         - hidden_dim - the size of RNN layer (number of hidden states)
         - vocab_size - size of vocabulary
         - p - dropout probability
    """

    def __init__(self, n_tokens, embedding_dim=128, hidden_dim=256, num_features=2048, p=0.5):

        super(DecoderRNN, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens
        # scale the inputs to softmax
        self.sample_temp = 0.5

        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(n_tokens, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        # produce the final output
        self.fc = nn.Linear(hidden_dim, n_tokens)

        # add attention layer
        self.attention = BahdanauAttention(num_features, hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)

    def forward(self, features, captions, sample_prob=0.0):

        """
        Arguments
        ----------
        - captions - image captions
        - features - features returned from Encoder
        - sample_prob - use it for scheduled sampling

        Returns
        ----------
        - outputs - output logits from t steps
        - atten_weights - weights from attention network
        """
        # create embeddings for captions of size (batch, sqe_len, embed_dim)
        features = features.unsqueeze(1)
        embed = self.embeddings(captions)
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.n_tokens)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size)

        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:, t, :]
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)

            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h, c))
            h = self.drop(h)
            output = self.fc(h)
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:, t, :] = output

            atten_weights[:, t, :] = atten_weight
        return outputs

    def init_hidden(self, features):
        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder

        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim=1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0


button_help = KeyboardButton('/help')
button_start = KeyboardButton('/start_captioning')
# button_monet = KeyboardButton('/Monet')
button_mscoco = KeyboardButton('/trained_with_mscoco')
button_mscoco_attn = KeyboardButton('/trained_with_mscoco_attention')
button_flickr8k = KeyboardButton('/trained_with_flickr8k')
button_flickr8k_attn = KeyboardButton('/trained_with_flickr8k_attention')

kb_help_and_start = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_help).add(button_start)
kb_help = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_help)
kb_choose_model = ReplyKeyboardMarkup(
    resize_keyboard=True, one_time_keyboard=True
).add(button_mscoco).add(button_mscoco_attn) \
    .add(button_flickr8k).add(button_flickr8k_attn).add(button_help)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class WaitForPic(StatesGroup):
    waiting_for_start = State()
    waiting_for_type_model = State()
    waiting_for_photo_mscoco = State()
    waiting_for_photo_mscoco_attn = State()
    waiting_for_photo_flickr8k = State()
    waiting_for_photo_flickr8k_attn = State()


@dp.message_handler(commands=['start', 'restart'], state="*")
async def process_start_command(message: types.Message):
    await message.reply(
        'Привет! Я умею генерировать описания к фото:) Чтобы попробовать нажми /start_captioning и пришли мне фотографию. '
        'Если возникнут вопросы, то просто нажми /help.', reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_start)
async def process_help_command(message: types.Message):
    await message.reply('Чтобы получить описание к фото нажми /start_captioning и пришли мне фото.',
                        reply_markup=kb_help_and_start)


@dp.message_handler(commands=['help'], state=WaitForPic.waiting_for_type_model)
async def get_transfer_type(message: types.Message):
    await message.reply('Выбери используемую модель из предложенных. Модели обучались на разных данных.',
                        reply_markup=kb_choose_model)
    await WaitForPic.waiting_for_type_model.set()


@dp.message_handler(commands=['help'],
                    state=[WaitForPic.waiting_for_photo_mscoco, WaitForPic.waiting_for_photo_mscoco_attn,
                           WaitForPic.waiting_for_photo_flickr8k, WaitForPic.waiting_for_photo_flickr8k_attn])
async def process_help_command(message: types.Message):
    await message.reply('Пришли мне фото, к которому мы будем генерировать описание.', reply_markup=kb_help)


@dp.message_handler(commands=['start_captioning'], state=WaitForPic.waiting_for_start)
async def get_transfer_type(message: types.Message):
    await message.reply('Выбери используемую модель из предложенных. Модели обучались на разных данных.',
                        reply_markup=kb_choose_model)
    await WaitForPic.waiting_for_type_model.set()


@dp.message_handler(commands=['trained_with_mscoco'], state=WaitForPic.waiting_for_type_model)
async def start_captioning(message: types.Message):
    await message.reply('Пришли мне фото, к которому мы будем генерировать описание.', reply_markup=kb_help)
    await WaitForPic.waiting_for_photo_mscoco.set()


@dp.message_handler(commands=['trained_with_mscoco_attention'], state=WaitForPic.waiting_for_type_model)
async def start_captioning(message: types.Message):
    await message.reply('Пришли мне фото, к которому мы будем генерировать описание.', reply_markup=kb_help)
    await WaitForPic.waiting_for_photo_mscoco_attn.set()


@dp.message_handler(commands=['trained_with_flickr8k'], state=WaitForPic.waiting_for_type_model)
async def start_captioning(message: types.Message):
    await message.reply('Пришли мне фото, к которому мы будем генерировать описание.', reply_markup=kb_help)
    await WaitForPic.waiting_for_photo_flickr8k.set()


@dp.message_handler(commands=['trained_with_flickr8k_attention'], state=WaitForPic.waiting_for_type_model)
async def start_captioning(message: types.Message):
    await message.reply('Пришли мне фото, к которому мы будем генерировать описание.', reply_markup=kb_help)
    await WaitForPic.waiting_for_photo_flickr8k_attn.set()


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_photo_mscoco)
async def get_content_photo_mscoco(message):
    photo_name = 'photo_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(photo_name)
    await message.reply('Отлично! Пожалуйста, подожди несколько минут, пока я генерирую описание...')
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    answer = image_captioning_model.generate_caption(photo_name, model_name='mscoco', t=5.0)

    await message.reply(' '.join(answer[1:-1]), reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()
    os.remove('photo_{}.jpg'.format(message.from_user.id))


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_photo_mscoco_attn)
async def get_content_photo_mscoco_attn(message):
    photo_name = 'photo_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(photo_name)
    await message.reply('Отлично! Пожалуйста, подожди несколько минут, пока я генерирую описание...')
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    answer = image_captioning_model.generate_caption(photo_name, model_name='mscoco_attn', t=5.0)

    await message.reply(' '.join(answer[1:-1]), reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()
    os.remove('photo_{}.jpg'.format(message.from_user.id))


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_photo_flickr8k)
async def get_content_photo_flickr8k(message):
    photo_name = 'photo_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(photo_name)
    await message.reply('Отлично! Пожалуйста, подожди несколько минут, пока я генерирую описание...')
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    answer = image_captioning_model.generate_caption(photo_name, model_name='flickr8k', t=5.0)

    await message.reply(' '.join(answer[1:-1]), reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()
    os.remove('photo_{}.jpg'.format(message.from_user.id))


@dp.message_handler(content_types=['photo'], state=WaitForPic.waiting_for_photo_flickr8k_attn)
async def get_content_photo_flickr8k_attn(message):
    photo_name = 'photo_{}.jpg'.format(message.from_user.id)
    await message.photo[0].download(photo_name)
    await message.reply('Отлично! Пожалуйста, подожди несколько минут, пока я генерирую описание...')
    await bot.send_chat_action(message.from_user.id, ChatActions.UPLOAD_DOCUMENT)
    answer = image_captioning_model.generate_caption(photo_name, model_name='flickr8k_attn', t=5.0)

    await message.reply(' '.join(answer[1:-1]), reply_markup=kb_help_and_start)
    await WaitForPic.waiting_for_start.set()
    os.remove('photo_{}.jpg'.format(message.from_user.id))


@dp.message_handler(content_types=ContentType.ANY, state="*")
async def unknown_message(msg: types.Message):
    message_text = 'Я не знаю, что мне делать! \nЕсли у тебя есть вопросы нажми кнопку /help ' \
                   'или начни сначала, нажав /restart.'
    await msg.reply(message_text, reply_markup=kb_help)


if __name__ == '__main__':
    executor.start_polling(dp)
