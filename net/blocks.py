import torch
import torch.nn as nn


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.sequence(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out=False):
        super(UpSampleBlock, self).__init__()
        sequence = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if drop_out:
            sequence += [nn.Dropout(0.5)]

        sequence += [nn.LeakyReLU(0.2)]
        self.sequence = nn.Sequential(*sequence)

    def forward(self, x):
        return self.sequence(x)


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, filters):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            DownSampleBlock(input_channels, filters),
            DownSampleBlock(filters, filters * 2),
            DownSampleBlock(filters * 2, filters * 4),
            DownSampleBlock(filters * 4, filters * 8),
            DownSampleBlock(filters * 8, filters * 8),
            DownSampleBlock(filters * 8, filters * 8),
            DownSampleBlock(filters * 8, filters * 8),
            DownSampleBlock(filters * 8, filters * 8),
        )
        self.decoder = nn.Sequential(
            UpSampleBlock(filters * 8, filters * 8, drop_out=True),
            UpSampleBlock(filters * 16, filters * 8, drop_out=True),
            UpSampleBlock(filters * 16, filters * 8, drop_out=True),
            UpSampleBlock(filters * 16, filters * 8, drop_out=True),
            UpSampleBlock(filters * 16, filters * 4),
            UpSampleBlock(filters * 8, filters * 2),
            UpSampleBlock(filters * 4, filters),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(filters * 2, output_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skip_conns = []
        for encode in self.encoder:
            x = encode(x)
            skip_conns.append(x)

        skip_conns = reversed(skip_conns[:-1])
        for decode, skip in zip(self.decoder, skip_conns):
            x = decode(x)
            x = torch.cat((x, skip), dim=1)

        x = self.last(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, filters):
        super(Discriminator, self).__init__()
        layers = 2
        sequence = [
            nn.Conv2d(input_channels, filters, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]

        mult = 1
        mult_prev = 1
        for n in range(1, layers):
            mult_prev, mult = mult, min(2**n, 8)
            sequence += [
                nn.Conv2d(filters * mult_prev, filters * mult, 4, 2, 1),
                nn.BatchNorm2d(filters * mult),
                nn.LeakyReLU(0.2),
            ]

        mult_prev, mult = mult, min(2**layers, 8)
        sequence += [
            nn.Conv2d(filters * mult_prev, filters * mult, 4, 1, 1),
            nn.BatchNorm2d(filters * mult),
            nn.LeakyReLU(0.2),
            nn.Conv2d(filters * mult, 1, 4, 1, 1),
            nn.Sigmoid(),
        ]

        self.sequence = nn.Sequential(*sequence)

    def forward(self, input):
        return self.sequence(input)
