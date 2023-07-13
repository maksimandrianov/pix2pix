import torch
import torch.nn as nn


class DownSampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DownSampleBlock, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, True),
        )

        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        skip_conn = self.sequence(x)
        x = self.pooling(skip_conn)
        return x, skip_conn


class UpSampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UpSampleBlock, self).__init__()
        self.up_sample = nn.ConvTranspose2d(input_channels, input_channels, 4, 2, 1, bias=False)

        self.sequence = nn.Sequential(
            nn.Conv2d(input_channels * 2, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x, skip_conn):
        x = self.up_sample(x)
        x = torch.cat([x, skip_conn], dim=1)
        x = self.sequence(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, filters):
        super(Generator, self).__init__()
        encoder = [
            DownSampleBlock(input_channels, filters),
        ]
        decoder = [
            UpSampleBlock(filters, output_channels),
        ]

        for i in range(3):
            lower_ = filters * 2**i
            higher_ = filters * 2 ** (i + 1)
            encoder.append(DownSampleBlock(lower_, higher_))
            decoder.append(UpSampleBlock(higher_, lower_))

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder[::-1])

    def forward(self, x):
        skip_conns = []
        for encode in self.encoder:
            x, skip_conn = encode(x)
            skip_conns += [skip_conn]

        for decode in self.decoder:
            x = decode(x, skip_conns.pop(-1))

        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, filters):
        super(Discriminator, self).__init__()
        layers = 3
        sequence = [
            nn.Conv2d(input_channels, filters, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
        ]

        mult = 1
        mult_prev = 1
        for n in range(1, layers):
            mult_prev, mult = mult, min(2**n, 8)
            sequence += [
                nn.Conv2d(filters * mult_prev, filters * mult, 4, 2, 1),
                nn.BatchNorm2d(filters * mult),
                nn.LeakyReLU(0.2, True),
            ]

        mult_prev, mult = mult, min(2**layers, 8)
        sequence += [
            nn.Conv2d(filters * mult_prev, filters * mult, 4, 1, 1),
            nn.BatchNorm2d(filters * mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(filters * mult, 1, 4, 1, 1),
        ]

        self.sequence = nn.Sequential(*sequence)

    def forward(self, input):
        return self.sequence(input)
