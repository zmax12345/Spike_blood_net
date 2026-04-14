import torch
import torch.nn as nn


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


# ==========================================
# 严格加回的 ANN 残差块 (Residual Block)
# ==========================================
class ANNResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ANNResidualBlock, self).__init__()
        # 保持通道数和分辨率不变，进行深度非线性融合
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = self.relu(out)
        return out


class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()

        # ==========================================
        # 在 CNN 解码层的入口处，插入 ANN 残差块
        # ==========================================
        self.res_block4 = ANNResidualBlock(512)  # 接收 SNN 第 4 层的特征
        self.res_block3 = ANNResidualBlock(256)  # 接收 SNN 第 3 层的特征
        self.res_block2 = ANNResidualBlock(128)  # 接收 SNN 第 2 层的特征
        self.res_block1 = ANNResidualBlock(64)  # 接收 SNN 第 1 层的特征

        # 转置卷积解码层
        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.softplus = nn.Softplus()

    def forward(self, skip_features, original_shape=(100, 368)):
        spk_sum_1, spk_sum_2, spk_sum_3, spk_sum_4 = skip_features

        # SNN 的特征先通过 ANN 残差块进行深度提炼，再参与后续的解码和拼接
        res_feat_4 = self.res_block4(spk_sum_4)
        res_feat_3 = self.res_block3(spk_sum_3)
        res_feat_2 = self.res_block2(spk_sum_2)
        res_feat_1 = self.res_block1(spk_sum_1)

        # 1. 瓶颈层特征通过残差块后，升维并进行第一次跳跃连接拼接
        d4 = self.up4(res_feat_4)
        d4_cropped = crop_like(d4, res_feat_3)
        cat3 = torch.cat((d4_cropped, res_feat_3), dim=1)

        # 2. 升维并进行第二次跳跃连接拼接
        d3 = self.up3(cat3)
        d3_cropped = crop_like(d3, res_feat_2)
        cat2 = torch.cat((d3_cropped, res_feat_2), dim=1)

        # 3. 升维并进行第三次跳跃连接拼接
        d2 = self.up2(cat2)
        d2_cropped = crop_like(d2, res_feat_1)
        cat1 = torch.cat((d2_cropped, res_feat_1), dim=1)

        # 4. 最后一次升维，输出去相关时间
        d1 = self.up1(cat1)

        if d1.shape[2:] != original_shape:
            d1 = d1[:, :, :original_shape[0], :original_shape[1]]

        tau_c = self.softplus(d1)
        return tau_c
