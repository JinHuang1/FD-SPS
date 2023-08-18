import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d


class Shared_Feature_extractor(nn.Module):
    def __init__(self):
        super(Shared_Feature_extractor, self).__init__()
        import torchvision.models as tmodels
        vgg16_s = tmodels.vgg16(pretrained=True)
        self.vgg16_s = nn.Sequential(*list(vgg16_s.children())[0][0:31])

    def forward(self, input):
        out = self.vgg16_s(input)

        return out


class Exclusive_Feature_extractor(nn.Module):
    def __init__(self):
        super(Exclusive_Feature_extractor, self).__init__()

        import torchvision.models as tmodels
        vgg16_s = tmodels.vgg16(pretrained=True)
        self.vgg16_s = nn.Sequential(*list(vgg16_s.children())[0][0:31])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        out = self.vgg16_s(input)
        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)

        return out


class Classifier(nn.Module):
    def __init__(self, feature_dim: int, output_dim, units: int = 256) -> None:
        """Simple dense classifier

        Args:
            feature_dim (int): [Number of input feature]
            output_dim ([type]): [Number of classes]
            units (int, optional): [Intermediate layers dimension]. Defaults to 15.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=feature_dim, out_channels=units, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=feature_dim, out_channels=units, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=feature_dim, out_channels=units, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=feature_dim, out_channels=units, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=4 * units, out_channels=units, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.units = units

        self.linear_sub = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, xs, ys, xe, ye):
        B, C, H, W = xs.size()
        P = H * W

        xs_c = self.conv1(xs)
        xs_t = xs_c.view(B, -1, P).permute(0, 2, 1)  # [B, HW, C]
        xe_c = self.conv3(xe)
        xe_t = xe_c.view(B, -1, P)  # [B, C, HW]
        x_matrix = F.softmax(torch.bmm(xs_t, xe_t), dim=-1)  # [B, HW, HW]

        ys_c = self.conv2(ys)
        ys_t = ys_c.view(B, -1, P).permute(0, 2, 1)  # [B, HW, C]
        ye_c = self.conv4(ye)
        ye_t = ye_c.view(B, -1, P)  # [B, C, HW]

        y_matrix = F.softmax(torch.bmm(ys_t, ye_t), dim=-1)

        # cross-modality global dependency weights
        weight_com = F.softmax(torch.mul(x_matrix, y_matrix), dim=-1)  # [B, HW, HW]

        x_dif = torch.cat([xs_c, xe_c, ys_c, ye_c], dim=1)
        x_dif_out = self.conv5(x_dif)

        x_dif_m = x_dif_out.view(B, -1, P)  # [B, C, HW]
        x_refine = torch.bmm(x_dif_m, weight_com).view(B, self.units, H, W)
        x_final = x_dif_out + x_refine

        x_dif = self.avgpool(x_final)
        x_dif = torch.flatten(x_dif, 1)
        x = self.linear_sub(x_dif)

        return x, x_final, [xs_c, ys_c, xe_c, ye_c]


class FDSPS(nn.Module):
    def __init__(self, nclass=2):
        super(FDSPS, self).__init__()
        # Encoders
        self.sh_enc = Shared_Feature_extractor()
        self.ex_enc = Exclusive_Feature_extractor()
        self.classifier = Classifier(feature_dim=512, output_dim=nclass)

    def forward(self, x, y):
        # Get the shared and exclusive features from x and y
        shared_x = self.sh_enc(x)
        shared_y = self.sh_enc(y)

        exclusive_x = self.ex_enc(x)
        exclusive_y = self.ex_enc(y)

        out, x_out, [shared_x, shared_y, exclusive_x, exclusive_y] = self.classifier(shared_x, shared_y, exclusive_x, exclusive_y)
        return out, x_out, [shared_x, shared_y, exclusive_x, exclusive_y]



