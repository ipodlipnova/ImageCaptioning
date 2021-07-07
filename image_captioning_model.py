import bz2
import pickle
from warnings import warn

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.model_zoo import load_url
from torchvision.models.inception import Inception3


class BeheadedInception3(Inception3):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        else:
            warn("Input isn't transformed")
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x_for_attn = x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x_for_capt = x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x_for_attn, x_for_capt, x


def beheaded_inception_v3(transform_input=True):
    model = BeheadedInception3(transform_input=transform_input, init_weights=True)
    inception_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    model.load_state_dict(load_url(inception_url))
    return model


inception = beheaded_inception_v3().train(False)


def generate_caption(image_name, model_name, caption_prefix=("#START#"),
                     t=1, sample=True, max_len=100):
    image = plt.imread(image_name)
    image = cv2.resize(src=image, dsize=(299, 299)).astype('float32') / 255.

    assert isinstance(image, np.ndarray) and np.max(image) <= 1 \
           and np.min(image) >= 0 and image.shape[-1] == 3

    if model_name == 'mscoco':
        with bz2.open('./models_pretrained/vocab_mscoco.pkl.bz2', 'rb') as f:
            vocab = pickle.load(f)
        with bz2.open('./models_pretrained/word_to_index_mscoco.pkl.bz2', 'rb') as f:
            word_to_index = pickle.load(f)
        with bz2.open('./models_pretrained/caption_net.pkl.bz2', 'rb') as f:
            network = pickle.load(f)

    elif model_name == 'mscoco_attn':
        with bz2.open('./models_pretrained/vocab_mscoco.pkl.bz2', 'rb') as f:
            vocab = pickle.load(f)
        with bz2.open('./models_pretrained/word_to_index_mscoco.pkl.bz2', 'rb') as f:
            word_to_index = pickle.load(f)
        with bz2.open('./models_pretrained/caption_net_attn.pkl.bz2', 'rb') as f:
            network = pickle.load(f)

    elif model_name == 'flickr8k':
        with bz2.open('./models_pretrained/vocab_flickr8.pkl.bz2', 'rb') as f:
            vocab = pickle.load(f)
        with bz2.open('./models_pretrained/word_to_index_flickr8.pkl.bz2', 'rb') as f:
            word_to_index = pickle.load(f)
        with bz2.open('./models_pretrained/caption_net_flickr8.pkl.bz2', 'rb') as f:
            network = pickle.load(f)

    elif model_name == 'flickr8k_attn':
        with bz2.open('./models_pretrained/vocab_flickr8.pkl.bz2', 'rb') as f:
            vocab = pickle.load(f)
        with bz2.open('./models_pretrained/word_to_index_flickr8.pkl.bz2', 'rb') as f:
            word_to_index = pickle.load(f)
        with bz2.open('./models_pretrained/caption_net_attn_flickr8.pkl.bz2', 'rb') as f:
            network = pickle.load(f)

    def as_matrix(sequences, max_len=None):
        """ Convert a list of tokens into a matrix with padding """
        eos_ix = word_to_index['#END#']
        unk_ix = word_to_index['#UNK#']
        pad_ix = word_to_index['#PAD#']

        max_len = max_len or max(map(len, sequences))

        matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
        for i, seq in enumerate(sequences):
            row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
            matrix[i, :len(row_ix)] = row_ix

        return matrix

    image = Variable(torch.FloatTensor(image.transpose([2, 0, 1])), volatile=True)

    vectors_8x8, vectors_neck, logits = inception(image[None])
    caption_prefix = [caption_prefix]

    for _ in range(max_len):

        prefix_ix = as_matrix([caption_prefix])
        prefix_ix = Variable(torch.LongTensor(prefix_ix), volatile=True)
        next_word_logits = network.forward(vectors_neck, prefix_ix)[0, -1]
        next_word_probs = F.softmax(next_word_logits, -1).data.numpy()

        assert len(next_word_probs.shape) == 1, 'probs must be one-dimensional'
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t)  # apply temperature

        if sample:
            next_word = np.random.choice(vocab, p=next_word_probs)
        else:
            next_word = vocab[np.argmax(next_word_probs)]

        caption_prefix.append(next_word)

        if next_word == "#END#":
            break
    print(caption_prefix)

    return caption_prefix