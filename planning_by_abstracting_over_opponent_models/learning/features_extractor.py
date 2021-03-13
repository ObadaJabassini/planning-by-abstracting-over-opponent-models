import math

import torch.nn as nn


class FeaturesExtractor(nn.Module):
    def __init__(self, image_size, conv_nb_layers, nb_filters, filter_size, filter_stride, filter_padding):
        super().__init__()
        self.conv = nn.Sequential()
        # We assume that image's width = image's height
        output_size = image_size
        for i in range(conv_nb_layers):
            input_shape = 1 if i == 0 else nb_filters
            self.conv.add_module(f"conv_{i + 1}", nn.Conv2d(in_channels=input_shape,
                                                            out_channels=nb_filters,
                                                            kernel_size=filter_size,
                                                            stride=filter_stride,
                                                            padding=filter_padding))
            self.conv.add_module(f"elu_{i + 1}", nn.ELU())
            output_size = int(math.floor((output_size - filter_size + 2 * filter_padding) / filter_stride)) + 1
        self.conv.add_module("flatten", nn.Flatten())
        self.output_size = output_size * output_size * nb_filters

    def forward(self, image):
        return self.conv(image)
