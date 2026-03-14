import torch
import torch.nn as nn
import torch.nn.functional as F


class PMNUNet(nn.Module):
    def __init__(self, in_nc: int = 1, out_nc: int = 1, nf: int = 32, res: bool = False):
        super().__init__()
        self.res = res

        self.conv1_1 = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(nf * 8, nf * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf * 16, nf * 16, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(nf * 16, nf * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf * 16, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf * 8, nf * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(nf * 4, nf * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(nf * 2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, w0 = x.shape[2:]
        x = self._pad_to_multiple(x, multiple=16)

        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)

        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)

        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)

        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))

        out = self.conv10_1(conv9)
        if self.res:
            out = out + x
        return out[:, :, :h0, :w0]
