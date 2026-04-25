import torch
import torch.nn as nn


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


# ==========================================
# 模块 1：时间通道注意力融合 (替代原先死板的 1x1 Conv)
# ==========================================
class TemporalAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(TemporalAttentionFusion, self).__init__()
        # Squeeze: 提取全局时间分布特征
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 对 10个分箱的 Mean 和 Std 进行动态打分
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()  # 输出 0~1 的注意力权重
        )
        # 融合与降维
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 1. 算出各通道打分
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 2. 动态加权（重要时间段保留，无效时间段抑制）
        x_attended = x * y.expand_as(x)
        # 3. 降维并激活
        return self.leaky_relu(self.conv1x1(x_attended))


# ==========================================
# 模块 2：环境自感知 Beta 预测器
# ==========================================
class BetaPredictor(nn.Module):
    def __init__(self, in_channels):
        super(BetaPredictor, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, 1),
            nn.Softplus()  # 保证算出来的 Beta 绝对是大于 0 的标量
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        beta = self.fc(y).view(b, 1, 1, 1)
        # 为了防止极端的 beta 导致梯度爆炸，给一个下限
        return beta + 1e-4


# ==========================================
# 模块 3：严格保留的 ANN 残差块
# ==========================================
class ANNResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ANNResidualBlock, self).__init__()
        # 我们按照您的要求，将这里的 BatchNorm 改为 InstanceNorm，消除 Batch=2 带来的训练集震荡
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class CNNDecoder(nn.Module):
    def __init__(self):
        super(CNNDecoder, self).__init__()

        multiplier = 20  # K=10 个 bin, Mean+Std

        # 引入带注意力机制的时序融合层
        self.fusion4 = TemporalAttentionFusion(512 * multiplier, 512)
        self.fusion3 = TemporalAttentionFusion(256 * multiplier, 256)
        self.fusion2 = TemporalAttentionFusion(128 * multiplier, 128)
        self.fusion1 = TemporalAttentionFusion(64 * multiplier, 64)

        # 引入可学习的 Beta 预测器 (直接从 SNN 的深层特征中感知环境光/偏振)
        self.beta_predictor = BetaPredictor(512 * multiplier)

        # 插入 ANN 残差块
        self.res_block4 = ANNResidualBlock(512)
        self.res_block3 = ANNResidualBlock(256)
        self.res_block2 = ANNResidualBlock(128)
        self.res_block1 = ANNResidualBlock(64)

        # 转置卷积解码层
        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.softplus = nn.Softplus()

    def forward(self, skip_features, original_shape=(100, 368)):
        feat_1, feat_2, feat_3, feat_4 = skip_features

        # ===== 核心：让网络自己推算出当前环境的 Beta 因子 =====
        beta = self.beta_predictor(feat_4)

        # 0. 注意力时序融合
        f4 = self.fusion4(feat_4)
        f3 = self.fusion3(feat_3)
        f2 = self.fusion2(feat_2)
        f1 = self.fusion1(feat_1)

        # 1. 残差块空间平滑
        res_feat_4 = self.res_block4(f4)
        res_feat_3 = self.res_block3(f3)
        res_feat_2 = self.res_block2(f2)
        res_feat_1 = self.res_block1(f1)

        # 2. 上采样与跳跃连接
        d4 = self.up4(res_feat_4)
        d4_cropped = crop_like(d4, res_feat_3)
        cat3 = torch.cat((d4_cropped, res_feat_3), dim=1)

        d3 = self.up3(cat3)
        d3_cropped = crop_like(d3, res_feat_2)
        cat2 = torch.cat((d3_cropped, res_feat_2), dim=1)

        d2 = self.up2(cat2)
        d2_cropped = crop_like(d2, res_feat_1)
        cat1 = torch.cat((d2_cropped, res_feat_1), dim=1)

        d1 = self.up1(cat1)

        if d1.shape[2:] != original_shape:
            d1 = d1[:, :, :original_shape[0], :original_shape[1]]

        # 输出原始的 tau_c
        raw_tau_c = self.softplus(d1)

        # ===== 核心物理校正 =====
        # 根据文献，SFI_corrected = Beta * SFI_raw。又因为 SFI ∝ 1/tau_c，所以 V ∝ Beta / tau_c
        # 为了不改动外面的代码 (V = d / tau_c_effective)，我们将有效 tau_c 设为 raw_tau_c / beta
        tau_c_effective = raw_tau_c / beta

        return tau_c_effective
