from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size * d - d) // 2
            self.blocks.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=d),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2),
            ))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


class MRF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rbs = nn.ModuleList([
            ResBlock(channels, 3, (1, 3, 5)),
            ResBlock(channels, 7, (1, 3, 5)),
            ResBlock(channels, 11, (1, 3, 5)),
        ])

    def forward(self, x):
        outs = [rb(x) for rb in self.rbs]
        return sum(outs) / len(outs)


class HiFiGANGenerator(nn.Module):
    def __init__(self, in_channels=80, upsample_rates=(8,8,2,2), upsample_kernel_sizes=(16,16,4,4), upsample_initial_channel=512):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        ch = upsample_initial_channel
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, kernel_size=k, stride=u, padding=(k-u)//2))
            ch = ch // 2
            self.mrfs.append(MRF(ch))
        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch, 1, 7, padding=3),
            nn.Tanh()
        )

    def forward(self, mel):
        x = self.conv_pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        return self.conv_post(x)
