from torch import nn
from affnet.modules.aff_block import FourierUnit
# ======================================================================================================================


class MS_CAM(nn.Module):


    def __init__(self, in_channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(in_channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x):
        xl = self.local_att(in_x)
        xg = self.global_att(in_x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return in_x * wei
# ======================================================================================================================


class AFF(nn.Module):


    def __init__(self, in_channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(in_channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x, input_y):
        xa = input_x + input_y
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * input_x * wei + 2 * input_y * (1 - wei)
        return xo
# ======================================================================================================================


class iAFF(nn.Module):


    def __init__(self, in_channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(in_channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x, input_y):
        xa = input_x + input_y
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = input_x * wei + input_y * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = input_x * wei2 + input_y * (1 - wei2)
        return xo
# ======================================================================================================================

class GF_LF(nn.Module):
    def __init__(self, in_channels=64, r=4):
        super(GF_LF, self).__init__()
        inter_channels = int(in_channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.fft_attn = FourierUnit(in_channels=in_channels, out_channels=in_channels, r=r)

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.fft_attn2 = FourierUnit(in_channels=in_channels, out_channels=in_channels, r=r)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x, input_y):
        xa = input_x + input_y  
        xl = self.local_att(xa)  
        xg = self.fft_attn(xa)  
        xlg = xl + xg  
        wei = self.sigmoid(xlg)
        xi = input_x * wei + input_y * (1 - wei)

        xl2 = self.fft_attn2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = input_x * wei2 + input_y * (1 - wei2)
        return xo
# ======================================================================================================================
# ======================================================================================================================



