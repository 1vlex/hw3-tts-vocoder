from torch import nn
import torch.nn.functional as F


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [16, 64, 256, 512, 1024]
        self.convs = nn.ModuleList()
        c = 1
        for o in chs:
            self.convs.append(nn.Conv1d(c, o, kernel_size=15, stride=2, padding=7))
            c = o
        self.conv_post = nn.Conv1d(c, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return x, feats


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3):
        super().__init__()
        self.ds = nn.ModuleList([ScaleDiscriminator() for _ in range(num_scales)])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        outs = []
        cur = x
        for i, d in enumerate(self.ds):
            if i > 0:
                cur = self.pool(cur)
            outs.append(d(cur))
        return outs


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), stride=(1, 1), padding=(2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), padding=(1, 0))

    def forward(self, x):
        B, C, T = x.shape
        if T % self.period != 0:
            pad = self.period - (T % self.period)
            x = F.pad(x, (0, pad), mode='reflect')
            T += pad
        x = x.view(B, C, T // self.period, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.conv_post(x)
        feats.append(x)
        return x, feats


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2,3,5,7,11)):
        super().__init__()
        self.ds = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        return [d(x) for d in self.ds]
