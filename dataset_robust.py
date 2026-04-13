import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import MinkowskiEngine as ME


class FlexibleBloodFlowDataset(Dataset):
    def __init__(self, data_config, mask_path="hot_pixel_mask.npy", T=1, seq_len=5000, dt_us=20):
        """
        data_config: 字典格式。支持传文件夹路径，也支持传特定 csv 文件路径。
                     例如 {"/data/zm/.../": 0.0105, "/data/zm/.../1.0mm_clip.csv": 0.0105}
        T: 每个切片的 bin 数量。为了保留 20us 分辨率，T 设为 1
        seq_len: 一次前向传播的总时间步数 (例如 100ms 对应 5000 步)
        dt_us: 基础时间间隔 20 微秒
        """
        self.data_config = data_config
        self.T = T
        self.seq_len = seq_len
        self.dt = dt_us

        self.hot_mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((800, 1280), dtype=bool)
        self.samples = self._build_dataset()

    def _build_dataset(self):
        samples = []
        for path, d_val in self.data_config.items():
            # 兼容文件夹或单个特定的 CSV 文件
            if os.path.isdir(path):
                csv_files = glob.glob(os.path.join(path, "*_clip.csv"))
            elif os.path.isfile(path) and path.endswith('.csv'):
                csv_files = [path]
            else:
                continue

            for file in csv_files:
                filename = os.path.basename(file)
                try:
                    # 针对 "1.0mm_clip.csv" 提取真值 1.0
                    v_true = float(filename.split('mm')[0])
                except ValueError:
                    print(f"警告: 无法从文件名 {filename} 提取流速，已跳过。")
                    continue

                # 严格按照原版 Sparse-PINN 的 pandas 读取与清洗逻辑
                try:
                    df = pd.read_csv(file, header=None, names=['row', 'col', 't_in', 't_off'],
                                     dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64, 't_off': np.int64},
                                     on_bad_lines='skip')
                except Exception as e:
                    print(f"警告: 读取 {filename} 失败，可能文件损坏。错误: {e}")
                    continue

                if df.empty: continue

                # 限制 ROI
                df = df[(df['row'] >= 400) & (df['row'] <= 499) & (df['col'] >= 200) & (df['col'] <= 567)].copy()
                if df.empty: continue

                # 过滤坏点
                valid_events = ~self.hot_mask[df['row'].values, df['col'].values]
                df = df[valid_events].copy()
                if df.empty: continue

                # 坐标平移，匹配网络输入尺寸 100x368
                df['row'] = df['row'] - 400
                df['col'] = df['col'] - 200

                # 时间对齐与量化
                t_start = df['t_in'].min()
                df['t_bin'] = (df['t_in'] - t_start) // self.dt

                max_bin = df['t_bin'].max()
                total_frames = int(max_bin // self.T) + 1

                # 提取时间切片序列
                for seq_start_idx in range(total_frames - self.seq_len + 1):
                    sequence_data = []
                    start_bin = seq_start_idx * self.T
                    end_bin = (seq_start_idx + self.seq_len) * self.T

                    seq_df = df[(df['t_bin'] >= start_bin) & (df['t_bin'] < end_bin)]

                    for f_idx in range(self.seq_len):
                        frame_start_bin = start_bin + f_idx * self.T
                        frame_df = seq_df[
                            (seq_df['t_bin'] >= frame_start_bin) & (seq_df['t_bin'] < frame_start_bin + self.T)]

                        if len(frame_df) == 0:
                            coords, feats = torch.empty((0, 3), dtype=torch.int32), torch.empty((0, 1),
                                                                                                dtype=torch.float32)
                        else:
                            locations = torch.IntTensor(
                                np.column_stack((frame_df['row'].values, frame_df['col'].values)))
                            features = torch.ones((len(frame_df), 1), dtype=torch.float32)
                            coords, feats = ME.utils.sparse_quantize(coordinates=locations, features=features,
                                                                     quantization_size=[1, 1])

                        sequence_data.append((coords, feats))

                    # 存入样本: (稀疏张量序列, 流速真值, 专属散斑尺寸)
                    samples.append((sequence_data, v_true, d_val))

                    # 避免对同一文件提取过多高度重合的长序列，每个 CSV 只截取第一段完整的 100ms 用于训练即可
                    break

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def sequence_sparse_collate(batch):
    seq_len = len(batch[0][0])
    batched_seq_data = []

    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        b_coords, b_feats = ME.utils.sparse_collate(coords_t, feats_t)
        batched_seq_data.append((b_coords, b_feats))

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    d_values = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)

    return batched_seq_data, labels, d_values