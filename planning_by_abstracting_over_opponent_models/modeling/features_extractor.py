import torch.nn as nn


class FeaturesExtractor(nn.Module):
    def __init__(self, image_size, conv_nb_layers, nb_filters, filter_size, filter_stride, filter_padding):
        super().__init__()
        conv_layers = []
        # We assume that image's width = image's height
        output_size = image_size
        for i in range(conv_nb_layers):
            temp = 3 if i == 0 else nb_filters
            conv_layers.append(nn.Conv2d(in_channels=temp,
                                         out_channels=nb_filters,
                                         kernel_size=filter_size,
                                         stride=filter_stride,
                                         padding=filter_padding))
            conv_layers.append(nn.ELU())
            output_size = (output_size - filter_size + 2 * filter_padding) / filter_stride + 1
        self.conv = nn.Sequential(*conv_layers, nn.Flatten())
        self.output_size = output_size * output_size * nb_filters

    def forward(self, image):
        return self.conv(image)
