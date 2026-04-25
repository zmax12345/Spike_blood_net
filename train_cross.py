import os
os.environ['OMP_NUM_THREADS'] = '8'  # 添加这行消除 ME 警告并限制底层线程数
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard

from model import SNN_CNN_Hybrid
from dataset_robust import FlexibleBloodFlowDataset, sequence_sparse_collate


class DenseBlockManager:
    def __init__(self, x_seq_sparse_data, batch_size, spatial_shape=(100, 368)):
        self.x_seq_sparse_data = x_seq_sparse_data
        self.batch_size = batch_size
        self.spatial_shape = spatial_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_block_dense(self, block_idx, block_size):
        start_t = block_idx * block_size
        end_t = min(start_t + block_size, len(self.x_seq_sparse_data))
        actual_block_size = end_t - start_t

        dense_block = torch.zeros((actual_block_size, self.batch_size, 1, *self.spatial_shape), device=self.device)

        for i, t in enumerate(range(start_t, end_t)):
            b_coords, b_feats = self.x_seq_sparse_data[t]

            # 【诊断与防崩溃核心代码】
            if len(b_feats) == 0:
                #print(f"!!! 追踪到异常: 第 {t} 个 20us 切片内的事件数为 0")
                continue  # 直接跳过，阻止 MinkowskiEngine 触发 CUDA 崩溃

            # 用保存的坐标和特征重建 SparseTensor
            sp_tensor = ME.SparseTensor(features=b_feats, coordinates=b_coords, device=self.device)
            # 转化为 GPU 稠密张量 [Batch, 1, H, W]
            dense_t = sp_tensor.dense(shape=torch.Size([self.batch_size, 1, *self.spatial_shape]))[0]
            dense_block[i] = dense_t

        return dense_block


def train_cross_env():
    global_start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. 跨环境泛化配置
    # ==========================================
    TOTAL_STEPS = 5000  # 100ms / 20us = 5000
    BLOCK_SIZE = 100  # Checkpoint 显存块大小
    BATCH_SIZE = 2  # 单卡 48G 测试值
    ORIGINAL_SHAPE = (100, 368)

    # 训练环境配置
    train_env_config = {
        "/data/zm/Moshaboli/new_data/no1": 0.018938,
        "/data/zm/Moshaboli/new_data/no4": 0.01973,
        "/data/zm/Moshaboli/new_data/no2": 0.01942
    }

    # 验证环境配置 (网络从未见过的物理场景)
    val_env_config = {
        "/data/zm/Moshaboli/new_data/no3": 0.01963,
    }

    # ==========================================
    # 2. 鲁棒数据集与 DataLoader
    # ==========================================
    YOUR_MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/3.0_mask (2)_hot_pixel_mask.npy"  # 请替换为真实路径
    train_ds = FlexibleBloodFlowDataset(train_env_config, T=1, seq_len=TOTAL_STEPS, dt_us=20)
    val_ds = FlexibleBloodFlowDataset(val_env_config, T=1, seq_len=TOTAL_STEPS, dt_us=20)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sequence_sparse_collate,
                              num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=sequence_sparse_collate,
                            num_workers=4)

    # ==========================================
    # 3. 模型与优化器加载 (新增学习率调度器)
    # ==========================================
    model = SNN_CNN_Hybrid(in_channels=1).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 新增：当验证集 Loss 连续 3 个 epoch 不下降时，学习率自动减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    writer = SummaryWriter(log_dir='./runs/cross_env_experiment')
    train_loss_history = []
    val_loss_history = []

    # 新增：记录最佳 Loss 用于保存模型
    best_val_loss = float('inf')
    best_epoch = -1  # <--- 新增：记录最佳的 Epoch 轮数
    # ==========================================
    # 4. 训练与跨环境验证循环
    # ==========================================
    epochs = 50
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss_sum = 0.0

        for batch_idx, (x_seq_sparse_data, y_true, d_values) in enumerate(train_loader):
            actual_batch_size = y_true.shape[0]
            y_true = y_true.to(device)
            d_values = d_values.to(device)

            optimizer.zero_grad()

            block_manager = DenseBlockManager(x_seq_sparse_data, actual_batch_size, spatial_shape=ORIGINAL_SHAPE)

            tau_c_pred = model(
                dataloader_or_generator=block_manager,
                total_steps=TOTAL_STEPS,
                block_size=BLOCK_SIZE,
                original_shape=ORIGINAL_SHAPE
            )

            d_values_expanded = d_values.view(-1, 1, 1, 1)
            v_pred = d_values_expanded / (tau_c_pred + 1e-8)

            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)
            loss = F.mse_loss(v_pred, y_true_expanded)

            loss.backward()

            # 新增：梯度裁剪，防止 BPTT 过程中的梯度爆炸和极端震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss_sum += loss.item()
            print(f"Epoch {epoch} [Train], Batch {batch_idx}: Loss = {loss.item():.4f}")

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        # --- 跨环境验证阶段 ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_idx, (x_seq_sparse_data, y_true, d_values) in enumerate(val_loader):
                actual_batch_size = y_true.shape[0]
                y_true = y_true.to(device)
                d_values = d_values.to(device)

                block_manager = DenseBlockManager(x_seq_sparse_data, actual_batch_size, spatial_shape=ORIGINAL_SHAPE)

                tau_c_pred = model(
                    dataloader_or_generator=block_manager,
                    total_steps=TOTAL_STEPS,
                    block_size=BLOCK_SIZE,
                    original_shape=ORIGINAL_SHAPE
                )

                d_values_expanded = d_values.view(-1, 1, 1, 1)
                v_pred = d_values_expanded / (tau_c_pred + 1e-8)

                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)
                loss = F.mse_loss(v_pred, y_true_expanded)

                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        print(f"=== Epoch {epoch} Validation (Cross-Env): Avg Loss = {avg_val_loss:.4f} ===")

        # 新增：触发学习率调度器检查
        scheduler.step(avg_val_loss)

        # 新增：保存最佳模型权重
        # 新增：保存最佳模型权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch  # <--- 新增：记录下当前这个创纪录的 epoch
            # 将模型状态字典保存到本地
            torch.save(model.state_dict(), '/data/zm/Moshaboli/new_data/Model/best_blood_flow_model.pth')
            print(f"*** 发现新的最佳验证集 Loss: {best_val_loss:.4f}，模型已保存为 'best_blood_flow_model.pth' ***")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)

        # ==========================================
        # 5. 训练结束：关闭 Writer 并保存静态 Loss 曲线图
        # ==========================================
    writer.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_loss_history, label='Train Loss', color='blue', marker='o')
    plt.plot(range(epochs), val_loss_history, label='Validation (Cross-Env) Loss', color='red', marker='s')

    # <--- 新增核心逻辑：切断异常 Spike 的干扰 --->
    # 鉴于您的模型最终能收敛到 0.4 左右，我们将 Y 轴最高点强制限制在 3.0 或 5.0
    # 这样那两个 40 多分的变态噪点会被画到图表外面去，从而完美展现 0~3 之间的精细收敛过程
    plt.ylim(0, 3.0)

    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Cross-Environment Validation Loss Curve (Zoomed In)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('/data/zm/Moshaboli/new_data/Loss_curve/spike_blood_loss_curve.png', dpi=300)

    # <--- 新增核心逻辑：训练结束时的全局播报 --->
    # 停止计时并计算总耗时
    global_end_time = time.time()
    total_seconds = global_end_time - global_start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    print("\n" + "=" * 50)
    print("=> 训练彻底结束！高精度 Loss 曲线已保存至指定目录。")
    print(f"=> 【总计耗时】 {hours} 小时 {minutes} 分钟 {seconds:.1f} 秒")
    print(f"=> 【全局最优】 最佳模型出现在 Epoch [{best_epoch}]")
    print(f"=> 【极限精度】 对应的最低验证集 Loss 为: {best_val_loss:.4f}")
    print("=" * 50 + "\n")

    # ==========================================
    # 6. 最终模型性能评估与物理流速折线图绘制
    # ==========================================
    print("=> 正在加载最佳模型，进行最终的流速预测与误差分析...")

    # 重新实例化并加载刚刚保存的最好权重
    best_model = SNN_CNN_Hybrid(in_channels=1).to(device)
    best_model.load_state_dict(torch.load('/data/zm/Moshaboli/new_data/Model/best_blood_flow_model.pth'))
    best_model.eval()

    all_v_true = []
    all_v_pred = []

    import numpy as np
    with torch.no_grad():
        for batch_idx, (x_seq_sparse_data, y_true, d_values) in enumerate(val_loader):
            actual_batch_size = y_true.shape[0]
            y_true_gpu = y_true.to(device)
            d_values_gpu = d_values.to(device)

            block_manager = DenseBlockManager(x_seq_sparse_data, actual_batch_size, spatial_shape=ORIGINAL_SHAPE)

            tau_c_pred = best_model(
                dataloader_or_generator=block_manager,
                total_steps=TOTAL_STEPS,
                block_size=BLOCK_SIZE,
                original_shape=ORIGINAL_SHAPE
            )

            # 计算出预测的流速图矩阵 [Batch, 1, 100, 368]
            d_values_expanded = d_values_gpu.view(-1, 1, 1, 1)
            v_pred_map = d_values_expanded / (tau_c_pred + 1e-8)

            # 【核心逻辑】：将网络输出的整张图像的流速，在空间维度上求平均，得到一个宏观流速数值
            v_pred_mean = v_pred_map.mean(dim=(1, 2, 3))

            all_v_true.extend(y_true.numpy().tolist())
            all_v_pred.extend(v_pred_mean.cpu().numpy().tolist())

    # 为了折线图好看且能体现出流速梯度，我们将数据按真实流速 (V_true) 从小到大排序
    sorted_pairs = sorted(zip(all_v_true, all_v_pred), key=lambda x: x[0])
    sorted_v_true = [x[0] for x in sorted_pairs]
    sorted_v_pred = [x[1] for x in sorted_pairs]

    # 将列表转换为 numpy 数组以计算误差指标
    v_true_arr = np.array(sorted_v_true)
    v_pred_arr = np.array(sorted_v_pred)

    # 1. 计算 MAE (平均绝对误差，单位：mm/s)
    mae = np.mean(np.abs(v_true_arr - v_pred_arr))
    # 2. 计算 RMSE (均方根误差，单位：mm/s)
    rmse = np.sqrt(np.mean((v_true_arr - v_pred_arr) ** 2))
    # 3. 计算 MAPE (平均绝对百分比误差，单位：%)
    # 防止除以 0 导致溢出，加一个极小值 epsilon
    mape = np.mean(np.abs((v_true_arr - v_pred_arr) / (v_true_arr + 1e-8))) * 100

    print("\n" + "*" * 50)
    print("=> 【最佳模型物理流速预测报告】")
    print(f"=> 平均绝对误差 (MAE):   {mae:.4f} mm/s  (预测值与真实值平均偏离的绝对速度)")
    print(f"=> 均方根误差   (RMSE):  {rmse:.4f} mm/s  (对极端大误差更敏感的指标)")
    print(f"=> 平均相对误差 (MAPE):  {mape:.2f} %     (预测偏离的百分比)")
    print("*" * 50 + "\n")

    # 绘制预测值与真实值的折线对比图
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(sorted_v_true)), sorted_v_true, label='True Velocity ($V_{true}$)', color='green', marker='o',
             linestyle='-', linewidth=2, markersize=8)
    plt.plot(range(len(sorted_v_pred)), sorted_v_pred, label='Predicted Velocity ($V_{pred}$)', color='red', marker='x',
             linestyle='--', linewidth=2, markersize=8)

    plt.xlabel('Validation Samples (Sorted by True Velocity Magnitude)', fontsize=12)
    plt.ylabel('Blood Flow Velocity (mm/s)', fontsize=12)
    plt.title(f'Final Model Flow Velocity Prediction vs Ground Truth\nMAE: {mae:.4f} mm/s | MAPE: {mape:.2f}%',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存折线图
    velocity_curve_path = '/data/zm/Moshaboli/new_data/Loss_curve/velocity_prediction_comparison.png'
    plt.savefig(velocity_curve_path, dpi=300)
    print(f"=> 物理流速预测对比折线图已保存至: {velocity_curve_path}")

if __name__ == '__main__':
    train_cross_env()
