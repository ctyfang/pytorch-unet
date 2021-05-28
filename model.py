import torch
from torch import nn


class SequentialConv(nn.Module):
    def __init__(self, c_in, c_out, n_convs=2, kernel=3):
        super(SequentialConv, self).__init__()
        self.layers = []
        for i_conv in range(n_convs):
            self.layers.append(nn.Conv2d(c_in, c_out, kernel))
            self.layers.append(nn.BatchNorm2d(c_out))
            self.layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, kernel=3, levels=4, c0=64, channel_factor=2):
        super(Encoder, self).__init__()
        self.convs = []
        self.max_pools = []
        self.levels = levels

        c_last = c0
        for i_level in range(1, self.levels+1):
            c_in, c_out = c_last, int(c0 * (channel_factor**i_level))
            self.convs.append(SequentialConv(c_in, c_out, kernel=kernel))
            self.max_pools.append(nn.MaxPool2d(2, (2, 2)))
            c_last = c_out

        self.convs = nn.ModuleList(self.convs)
        self.max_pools = nn.ModuleList(self.max_pools)

    def forward(self, x):
        x_intermediates = []
        for i_level in range(self.levels):
            x = self.convs[i_level](x)
            x_intermediates.append(x)
            x = self.max_pools[i_level](x)

        return x, x_intermediates


class Decoder(nn.Module):
    def __init__(self, kernel=3, levels=4, c0=1024, channel_factor=0.5):
        super(Decoder, self).__init__()
        self.convs = []
        self.up_convs = []
        self.levels = levels

        c_last = c0
        for i_level in range(1, self.levels + 1):
            c_in, c_out = c_last, int(c0 * (channel_factor**i_level))
            self.up_convs.append(nn.ConvTranspose2d(c_in, c_in//2, 2, (2, 2)))
            self.convs.append(SequentialConv(c_in, c_out, kernel=kernel))
            c_last = c_out

        self.up_convs = nn.ModuleList(self.up_convs)
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x, x_intermediates):
        for i_level in range(self.levels):

            # Up-Conv and apply Skip-Connection
            x = self.up_convs[i_level](x)
            x_skip = x_intermediates[-1-i_level]
            x_skip = torch.nn.functional.interpolate(x_skip, size=x.shape[2:])
            x = torch.cat([x, x_skip], dim=1)

            # Conv
            x = self.convs[i_level](x)

        return x


class UNet(nn.Module):
    def __init__(self, n_classes=1, n_prefeats=64):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.n_prefeats = n_prefeats

        self.pre_encoder = SequentialConv(1, n_prefeats, 2, 3)
        self.pre_down = nn.MaxPool2d(2, (2, 2))
        self.encoder = Encoder(levels=3)

        self.n_bottleneck_feats = int(n_prefeats * (2**(self.encoder.levels+1)))
        self.bottleneck = SequentialConv(self.n_bottleneck_feats//2,
                                         self.n_bottleneck_feats)

        self.decoder = Decoder(levels=4)

        self.n_output_feats = int(self.n_bottleneck_feats * (0.5**(self.decoder.levels)))
        self.output = nn.Conv2d(self.n_output_feats, self.n_classes, kernel_size=1)

    def forward(self, x):
        x = self.pre_down(self.pre_encoder(x))
        skip_connections = [x]

        x, x_intermed = self.encoder(x)
        skip_connections += x_intermed

        x = self.bottleneck(x)

        x = self.decoder(x, skip_connections)
        return self.output(x)