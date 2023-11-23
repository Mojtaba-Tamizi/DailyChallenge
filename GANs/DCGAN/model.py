import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d) -> None:
        super(Discriminator, self).__init__()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        