import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from snn_encoder import SNNEncoder  # 导入第一阶段
from cnn_decoder import CNNDecoder  # 导入第三阶段


class SNN_CNN_Hybrid(nn.Module):
    def __init__(self, in_channels=1):
        super(SNN_CNN_Hybrid, self).__init__()
        # 实例化编码器和解码器
        self.snn_encoder = SNNEncoder(in_channels=in_channels)
        self.cnn_decoder = CNNDecoder()

    def forward(self, dataloader_or_generator, total_steps=5000, block_size=100, original_shape=(100, 368)):
        mems = (None, None, None, None)

        # ==========================================
        # 新增：时序分组(Binning)参数设计
        # ==========================================
        K = 10  # 将 5000 步分成 10 个时间段，每个时间段 500 步 (10ms)
        steps_per_bin = total_steps // K
        blocks_per_bin = steps_per_bin // block_size  # 每个 Bin 包含 5 个 Block

        # 当前 Bin 的膜电位求和与平方求和，用于计算均值(Mean)和标准差(Std)
        bin_m_sum = [0, 0, 0, 0]
        bin_m_sq = [0, 0, 0, 0]

        # 存放最终 4 层的分组特征 (每个 List 会装入 20 个 Tensor: 10个Mean + 10个Std)
        layer_features = [[], [], [], []]

        num_blocks = total_steps // block_size

        # --- 阶段二：时间步分块与静态物理特征提取 ---
        for b in range(num_blocks):
            x_block = dataloader_or_generator.get_block_dense(b, block_size).cuda()

            def block_forward(x_blk, m1, m2, m3, m4):
                local_mems = (m1, m2, m3, m4)
                # 局部累加器
                m1_sum, m2_sum, m3_sum, m4_sum = 0, 0, 0, 0
                m1_sq, m2_sq, m3_sq, m4_sq = 0, 0, 0, 0

                for t in range(block_size):
                    x_t = x_blk[t]
                    spks, local_mems = self.snn_encoder.forward_step(x_t, local_mems)

                    # 累加膜电位 (Mean所需)
                    m1_sum = m1_sum + local_mems[0]
                    m2_sum = m2_sum + local_mems[1]
                    m3_sum = m3_sum + local_mems[2]
                    m4_sum = m4_sum + local_mems[3]

                    # 累加膜电位的平方 (Std所需)
                    m1_sq = m1_sq + local_mems[0] ** 2
                    m2_sq = m2_sq + local_mems[1] ** 2
                    m3_sq = m3_sq + local_mems[2] ** 2
                    m4_sq = m4_sq + local_mems[3] ** 2

                return local_mems, (m1_sum, m2_sum, m3_sum, m4_sum), (m1_sq, m2_sq, m3_sq, m4_sq)

            out = checkpoint(block_forward, x_block, *mems, use_reentrant=False)
            mems, loc_m_sum, loc_m_sq = out[0], out[1], out[2]

            # 将本 Block 的局部累加值，汇入当前 Bin 的统计中
            for i in range(4):
                bin_m_sum[i] = bin_m_sum[i] + loc_m_sum[i]
                bin_m_sq[i] = bin_m_sq[i] + loc_m_sq[i]

            # 检查当前 Bin (即凑满 500 步) 是否已结束
            if (b + 1) % blocks_per_bin == 0:
                for i in range(4):
                    # 1. 结算均值 (Mean)
                    mean = bin_m_sum[i] / steps_per_bin
                    # 2. 结算方差 = E(X^2) - (E(X))^2
                    variance = (bin_m_sq[i] / steps_per_bin) - (mean ** 2)
                    # 强制消除浮点数极小误差导致的负方差崩溃
                    variance = torch.clamp(variance, min=1e-8)
                    # 3. 结算标准差 (Std)
                    std = torch.sqrt(variance)

                    # 收集该时间段的两大物理特征
                    layer_features[i].append(mean)
                    layer_features[i].append(std)

                # 重置 Bin 累加器，准备迎接下一个 10ms
                bin_m_sum = [0, 0, 0, 0]
                bin_m_sq = [0, 0, 0, 0]

            del x_block
            torch.cuda.empty_cache()

        # 在通道维度拼接 10 个时间段的 Mean 和 Std (K=10 * 2 = 20 倍原通道)
        cat_feat_1 = torch.cat(layer_features[0], dim=1)
        cat_feat_2 = torch.cat(layer_features[1], dim=1)
        cat_feat_3 = torch.cat(layer_features[2], dim=1)
        cat_feat_4 = torch.cat(layer_features[3], dim=1)

        skip_features = (cat_feat_1, cat_feat_2, cat_feat_3, cat_feat_4)

        # --- 阶段三：静态 CNN 解码与输出 ---
        tau_c = self.cnn_decoder(skip_features, original_shape=original_shape)

        return tau_c
