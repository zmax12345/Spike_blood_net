import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# 继承自 Sparse-PINN 的替代梯度函数，用于反向传播
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=3.0):
        ctx.scale = scale
        ctx.save_for_backward(input)
        output = torch.zeros_like(input, dtype=input.dtype)
        output[input > 0] = 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        return grad, None


# 结合 Spike-FlowNet 的稠密卷积与 Sparse-PINN 的多阈值发放 (MSF)
class DenseMSFConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(DenseMSFConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 基础特征提取：标准 2D 卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # MSF 神经元参数
        self.D = 4  # 最大突触连接数（单步最多激发 4 个脉冲）
        self.h = 1.0  # 阈值间隔
        self.beta = nn.Parameter(torch.tensor([0.8]))
        self.b = nn.Parameter(torch.tensor([0.1]))  # 初始阈值 V_th

        self.spike_fn = SurrogateHeaviside.apply
        self.eps = 1e-8

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.normal_(self.beta, mean=0.8, std=0.01)
        torch.nn.init.normal_(self.b, mean=0.1, std=0.01)
        torch.nn.init.xavier_uniform_(self.conv.weight, torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, x, mem):
        # x: [Batch, Channels, H, W] - 某一特定微秒级的稠密切片
        conv_out = self.conv(x)

        if mem is None:
            mem = torch.zeros_like(conv_out)

        # 膜电位泄漏与积分 (Leaky & Integrate)
        new_mem = mem * self.beta + conv_out * (1. - self.beta)

        # 维度对齐以进行广播减法 (U - V_th)
        b = self.b.view(1, -1, 1, 1)
        mthr = new_mem - b

        # 【向量化优化后】：一次性并行计算所有阈值，彻底消灭 Python for 循环
        # 创建维度为 [4, 1, 1, 1, 1] 的阈值倍数张量
        d_vals = torch.arange(self.D, device=mthr.device, dtype=mthr.dtype).view(self.D, 1, 1, 1, 1)

        # 将 mthr 扩展为 [1, Batch, Channels, Height, Width]
        mthr_expanded = mthr.unsqueeze(0)

        # 利用广播机制一次性减去所有阈值：得到 [4, Batch, Channels, Height, Width]
        mthr_all = mthr_expanded - d_vals * self.h

        # 一次性调用 spike_fn（替代了原来的 4 次调用），并在第 0 维度求和压平
        spk = self.spike_fn(mthr_all).sum(dim=0)

        # 膜电位重置 (Hard Reset)：触发任何脉冲则归零
        spk_mask = (spk > 0).float()
        final_mem = new_mem * (1. - spk_mask)

        return spk, final_mem


# SNN 编码器模块：包含 4 层降维结构
class SNNEncoder(nn.Module):
    def __init__(self, in_channels=1):  # 根据输入极性决定，无极性设为1
        super(SNNEncoder, self).__init__()

        # 严格保留要求的通道扩增与 Stride=2 降维
        self.enc1 = DenseMSFConv2D(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.enc2 = DenseMSFConv2D(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = DenseMSFConv2D(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc4 = DenseMSFConv2D(256, 512, kernel_size=3, stride=2, padding=1)

    def forward_step(self, x_t, mems):
        """
        处理单个时间步的前向传播。
        将其独立成函数，是为了方便后续在 Block 级别使用 torch.utils.checkpoint
        """
        mem1, mem2, mem3, mem4 = mems

        spk1, mem1 = self.enc1(x_t, mem1)
        spk2, mem2 = self.enc2(spk1, mem2)
        spk3, mem3 = self.enc3(spk2, mem3)
        spk4, mem4 = self.enc4(spk3, mem4)

        # 返回当前时间步生成的脉冲和更新后的膜电位
        return (spk1, spk2, spk3, spk4), (mem1, mem2, mem3, mem4)
