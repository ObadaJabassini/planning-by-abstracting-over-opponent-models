import math

import torch.nn as nn


class FeaturesExtractor(nn.Module):
    def __init__(self, input_size, nb_filters, filter_size, filter_stride, filter_padding):
        super().__init__()
        output_width, output_height, input_channel = input_size
        conv = nn.Sequential()
        l = len(nb_filters)
        for i in range(l):
            conv.add_module(f"conv_{i + 1}", nn.Conv2d(in_channels=input_channel if i == 0 else nb_filters[i - 1],
                                                       out_channels=nb_filters[i],
                                                       kernel_size=filter_size,
                                                       stride=filter_stride,
                                                       padding=filter_padding))
            conv.add_module(f"elu_{i + 1}", nn.ELU())
            output_width = int(math.floor((output_width - filter_size + 2 * filter_padding) / filter_stride)) + 1
            output_height = int(math.floor((output_height - filter_size + 2 * filter_padding) / filter_stride)) + 1
        conv.add_module("flatten", nn.Flatten())
        self.conv = conv
        self.output_size = output_width * output_height * nb_filters[l - 1]

    def forward(self, image):
        return self.conv(image)
