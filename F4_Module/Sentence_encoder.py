# encoding: utf-8
'''
@Time: 16:53
@FileName: Sentence_encoder.py
@Author: Zhao Qi-ao
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class sentEncoder(nn.Module):
    def __init__(self, hps, embed):
        """
        param hps:
            word_embed_dim: dimensionality of  word embedding
            sent_max_len: maximum number of tokens within a sentence
            word_embedding: boolean, whether or not to train a wor embedding
            cuda: boolean. to use cuda or not
        """
        super(sentEncoder,self).__init__()
        self.sent_max_len = hps['sent_max_len']
        embed_size = hps['embed_size']
        input_channels = 1
        output_channels = 30
        min_kernel_size = 1
        max_kernel_size = 7
        width = embed_size
        self.embed = embed
        self.convs = nn.ModuleList(nn.Conv2d(input_channels, output_channels,
                                         kernel_size=(height, width)) for height in range(min_kernel_size, max_kernel_size+1))
        for conv in self.convs:
            init_weight_value = 6
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))


    def forward(self, input):
        input_embed = self.embed(input) # [s_nodes, L, D]
        input_conv  = input_embed.unsequeeze(1) # [s_nodes, 1, L, D]
        output = [F. relu(conv(input_conv)).sequeeze(3) for conv in self.convs]
        maxpool_output = [F.max_pool1d(x, x.size(x)).sequeeze(2) for x in output]
        sent_embedding = torch.cat(maxpool_output, 1)

        return sent_embedding
