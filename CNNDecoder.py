import torch
import torch.nn as nn


def crop_like(input, target):
    """
    尺寸对齐函数：解决降维再升维过程中，因奇数分辨率导致的像素偏差问题。
    将上采样后的特征图 (input) 裁剪到与跳跃连接特征图 (target) 完全相同的尺寸。
    """
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        # 保留左上角，裁剪掉多余的边缘像素
        return input[:, :, :target.size(2), :target.size(3)]


class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()

        # 解码层 4：接收 SNN 第 4 层的特征 (512)，上采样后准备与 SNN 第 3 层 (256) 拼接
        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        # 解码层 3：接收拼接后的特征 (256 + 256 = 512)，上采样后准备与 SNN 第 2 层 (128) 拼接
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)

        # 解码层 2：接收拼接后的特征 (128 + 128 = 256)，上采样后准备与 SNN 第 1 层 (64) 拼接
        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 解码层 1：接收拼接后的特征 (64 + 64 = 128)，最后一次上采样恢复至原图，输出 1 个通道(tau_c)
        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)

        # 保证去相关时间 tau_c 为正数
        self.softplus = nn.Softplus()

    def forward(self, skip_features, original_shape=(100, 367)):
        spk_sum_1, spk_sum_2, spk_sum_3, spk_sum_4 = skip_features

        # 1. 升维并进行第一次跳跃连接拼接
        d4 = self.up4(spk_sum_4)
        d4_cropped = crop_like(d4, spk_sum_3)
        cat3 = torch.cat((d4_cropped, spk_sum_3), dim=1)

        # 2. 升维并进行第二次跳跃连接拼接
        d3 = self.up3(cat3)
        d3_cropped = crop_like(d3, spk_sum_2)
        cat2 = torch.cat((d3_cropped, spk_sum_2), dim=1)

        # 3. 升维并进行第三次跳跃连接拼接
        d2 = self.up2(cat2)
        d2_cropped = crop_like(d2, spk_sum_1)
        cat1 = torch.cat((d2_cropped, spk_sum_1), dim=1)

        # 4. 最后一次升维，输出去相关时间
        d1 = self.up1(cat1)

        # 确保最终输出尺寸严格等于输入的原图分辨率 (100x367)
        if d1.shape[2:] != original_shape:
            d1 = d1[:, :, :original_shape[0], :original_shape[1]]

        tau_c = self.softplus(d1)
        return tau_c