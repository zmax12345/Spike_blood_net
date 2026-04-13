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

    def forward(self, dataloader_or_generator, total_steps=5000, block_size=100, original_shape=(100, 367)):
        mems = (None, None, None, None)
        spk_sum_1, spk_sum_2, spk_sum_3, spk_sum_4 = 0, 0, 0, 0
        num_blocks = total_steps // block_size

        # --- 阶段二：时间步分块与静态特征聚合 ---
        for b in range(num_blocks):
            # 获取当前 block 的 [block_size, Batch, C, H, W] 数据
            x_block = dataloader_or_generator.get_block_dense(b, block_size).cuda()

            def block_forward(x_blk, m1, m2, m3, m4):
                local_mems = (m1, m2, m3, m4)
                loc_sum1, loc_sum2, loc_sum3, loc_sum4 = 0, 0, 0, 0
                for t in range(block_size):
                    x_t = x_blk[t]
                    spks, local_mems = self.snn_encoder.forward_step(x_t, local_mems)
                    loc_sum1, loc_sum2 = loc_sum1 + spks[0], loc_sum2 + spks[1]
                    loc_sum3, loc_sum4 = loc_sum3 + spks[2], loc_sum4 + spks[3]
                return local_mems, (loc_sum1, loc_sum2, loc_sum3, loc_sum4)

            out = checkpoint(block_forward, x_block, *mems, use_reentrant=False)
            mems, local_sums = out[0], out[1]

            spk_sum_1 = spk_sum_1 + local_sums[0]
            spk_sum_2 = spk_sum_2 + local_sums[1]
            spk_sum_3 = spk_sum_3 + local_sums[2]
            spk_sum_4 = spk_sum_4 + local_sums[3]

            del x_block
            torch.cuda.empty_cache()

        skip_features = (spk_sum_1, spk_sum_2, spk_sum_3, spk_sum_4)

        # --- 阶段三：静态 CNN 解码与输出 ---
        tau_c = self.cnn_decoder(skip_features, original_shape=original_shape)

        return tau_c