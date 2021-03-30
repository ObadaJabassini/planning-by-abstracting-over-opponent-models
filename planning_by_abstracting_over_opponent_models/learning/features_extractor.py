import math

import torch.nn as nn


class FeaturesExtractor(nn.Module):
    def __init__(self, input_size, nb_conv_layers, nb_filters, filter_size, filter_stride, filter_padding):
        super().__init__()
        conv = nn.Sequential()
        output_width, output_height, input_channel = input_size
        for i in range(1, nb_conv_layers + 1):
            input_shape = input_channel if i == 1 else nb_filters
            conv.add_module(f"conv_{i}", nn.Conv2d(in_channels=input_shape,
                                                   out_channels=nb_filters,
                                                   kernel_size=filter_size,
                                                   stride=filter_stride,
                                                   padding=filter_padding))
            conv.add_module(f"elu_{i}", nn.ELU())
            output_width = int(math.floor((output_width - filter_size + 2 * filter_padding) / filter_stride)) + 1
            output_height = int(math.floor((output_height - filter_size + 2 * filter_padding) / filter_stride)) + 1
        conv.add_module("flatten", nn.Flatten())
        self.conv = conv
        self.output_size = output_width * output_height * nb_filters

    def forward(self, image):
        return self.conv(image)
