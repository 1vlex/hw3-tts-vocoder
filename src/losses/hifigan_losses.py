import torch
import torch.nn.functional as F


def discriminator_loss(real_outputs, fake_outputs):
    loss = 0.0
    for (dr, _), (df, _) in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((1.0 - dr) ** 2) + torch.mean(df ** 2)
    return loss


def generator_adv_loss(fake_outputs):
    loss = 0.0
    for (df, _) in fake_outputs:
        loss = loss + torch.mean((1.0 - df) ** 2)
    return loss


def feature_matching_loss(real_outputs, fake_outputs):
    loss = 0.0
    for (_, frs), (_, ffs) in zip(real_outputs, fake_outputs):
        for fr, ff in zip(frs, ffs):
            loss = loss + F.l1_loss(ff, fr.detach())
    return loss
