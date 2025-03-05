import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


# 参数设置
data_folder = "/home/wangy/code/EDITS复现/multi-view/sorted_disk_data"
output_folder = "/home/wangy/code/EDITS复现/multi-view-mine/fusion_feature"
raw_feature_output_folder='/home/wangy/code/EDITS复现/multi-view-mine/raw_feature'
histogram_feature_feature_output_folder='/home/wangy/code/EDITS复现/multi-view-mine/histogram_feature'
sequence_feature_output_folder='/home/wangy/code/EDITS复现/multi-view-mine/sequence_feature'
default_window_size = 256
step_size = 3
bucket_count = 100  # M，桶的数量
batch_size = 10000  # 每批次存储的记录数
mean_features_file = "/home/wangy/code/EDITS复现/multi-view/mean_sequence_features.csv"
selected_features_path = "/home/wangy/code/EDITS复现/multi-view/selected_histogram_features_bucket/selected_features_union_new.npy"
segments=4
min_max_path = "/home/wangy/code/EDITS复现/multi-view/min_max_values.npy"
selected_sequence_features_path='/home/wangy/code/EDITS复现/multi-view/selected_sequence_feature/selected_features_union_sequence.npy'
# 加载选中的直方图特征

selected_histogram_features = np.load(selected_features_path, allow_pickle=True)
selected_sequence_features = np.load(selected_sequence_features_path, allow_pickle=True)

print(f'hist{len(selected_histogram_features)}')
print(f'se{len(selected_sequence_features)}')
print(selected_sequence_features)

# 载入均值特征
mean_features = pd.read_csv(mean_features_file, header=None).values.flatten()

# 计算全局最小值和最大值
def calculate_min_max(data_folder):
    disk_files = [f for f in os.listdir(data_folder) if f.startswith("disk_disk_")]
    min_vals, max_vals = None, None

    for disk_file in tqdm(disk_files, desc="Calculating min and max values"):
        disk_data = pd.read_csv(os.path.join(data_folder, disk_file))
        disk_data = disk_data.drop(columns=['label', 'serial_number', 'model', 'dt'], errors='ignore')

        if min_vals is None:
            min_vals = disk_data.min().values
            max_vals = disk_data.max().values
        else:
            min_vals = np.minimum(min_vals, disk_data.min().values)
            max_vals = np.maximum(max_vals, disk_data.max().values)

    return min_vals, max_vals

# 生成桶区间
def create_buckets(min_vals, max_vals, bucket_count):
    buckets = []
    for min_val, max_val in zip(min_vals, max_vals):
        bucket_width = (max_val - min_val) / (bucket_count - 2) if max_val > min_val else 1
        bucket_edges = [min_val + i * bucket_width for i in range(bucket_count - 1)]
        buckets.append([-np.inf] + bucket_edges + [np.inf])
    return buckets

# 计算直方图特征
def calculate_histogram_features(window_data, buckets):
    histogram_features = []
    for feature_column, bucket_edges in zip(window_data.T, buckets):
        hist, _ = np.histogram(feature_column, bins=bucket_edges)
        histogram_features.extend(hist / len(feature_column))  # 归一化
    return histogram_features


# 计算原始特征
def extract_raw_features(data):
    """原始特征使用窗口的最后一条记录"""
    return data[-30:]

# 计算序列特征
def calculate_cvar(segment, mean_value):
    """计算变异系数 (CVAR)"""
    if len(segment) == 0 or mean_value == 0:
        return 0
    return np.sqrt(np.sum((segment - mean_value) ** 2) / len(segment)*segments) / mean_value


def calculate_kurt(segment, mean_value):
    """计算峰度 (KURT)"""
    if len(segment) <= 1:
        return 0
    m2 = np.sum((segment - mean_value) ** 2) / len(segment)
    m4 = np.sum((segment - mean_value) ** 4) / len(segment)
    return ((segments*m4) / (segments*m2) ** 2) - 3


def calculate_slope(segment, start_idx, end_idx):
    """计算斜率 (SLOPE)"""
    if len(segment) == 0 or end_idx - start_idx == 0:
        return 0
    return (segment[-1] - segment[0]) / (end_idx - start_idx)

def calculate_sequence_features(data, segments, mean_features):
    """计算每个段的 CVAR, KURT, SLOPE 特征"""
    segment_length = len(data) // segments
    features = []

    for i in range(segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < segments - 1 else len(data)
        segment = data[start:end]

        # 计算均值
        mean_value = np.mean(segment)*segments

        # 计算 CVAR, KURT 和 SLOPE
        cvar = calculate_cvar(segment, mean_value)
        kurt = calculate_kurt(segment, mean_value)
        slope = calculate_slope(segment, start, end)

        # 计算差异
        cvar_diff = cvar - mean_features[i * 3]  # 根据顺序调整
        kurt_diff = kurt - mean_features[i * 3 + 1]
        slope_diff = slope - mean_features[i * 3 + 2]

        # 将这些特征添加到列表中
        features.extend([cvar, kurt, slope, cvar_diff, kurt_diff, slope_diff])

    return features
# 保存融合特征
# 定义注意力模块
class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, time_steps):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.time_steps = time_steps

    def forward(self, raw_features, histogram_features, sequence_features):
        # 转换直方图和序列特征为与原始特征一致的形状 (time_steps, feature_dim)
        histogram_features = histogram_features.unsqueeze(0).repeat(self.time_steps, 1)
        sequence_features = sequence_features.unsqueeze(0).repeat(self.time_steps, 1)

        # 查询 (query), 键 (key), 值 (value)
        query = self.query(raw_features)  # shape: (time_steps, feature_dim)
        key = torch.stack([
            self.key(histogram_features),
            self.key(sequence_features),
            self.key(raw_features)
        ])  # shape: (3, time_steps, feature_dim)
        value = torch.stack([
            self.value(histogram_features),
            self.value(sequence_features),
            self.value(raw_features)
        ])  # shape: (3, time_steps, feature_dim)

        # 计算注意力权重
        attention_weights = F.softmax(torch.bmm(query.unsqueeze(0), key.transpose(1, 2)), dim=-1)  # shape: (1, time_steps, 3)
        fused_features = torch.bmm(attention_weights, value.transpose(1, 2)).squeeze(0)  # shape: (time_steps, feature_dim)

        return fused_features

# 主函数中插入注意力融合逻辑
def apply_attention_fusion(raw_features, histogram_features, sequence_features, time_steps):
    feature_dim = raw_features.shape[1]
    attention_fusion = AttentionFusion(feature_dim, time_steps)
    raw_features = torch.tensor(raw_features, dtype=torch.float32)
    histogram_features = torch.tensor(histogram_features, dtype=torch.float32)
    sequence_features = torch.tensor(sequence_features, dtype=torch.float32)

    fused_features = attention_fusion(raw_features, histogram_features, sequence_features)
    return fused_features

def save_batch(fusion_features, labels, batch_count):
    fusion_df = pd.DataFrame(fusion_features)
    fusion_df['label'] = labels
    output_file = os.path.join(output_folder, f"fusion_features_batch_{batch_count}.csv")
    fusion_df.to_csv(output_file, index=False)
    print(f"Batch {batch_count} saved: Fusion features to {output_file}")

def raw_feature_save_batch(raw_features, labels, batch_count):
    fusion_df = pd.DataFrame(raw_features)
    fusion_df['label'] = labels
    output_file = os.path.join(output_folder, f"raw_features_batch_{batch_count}.csv")
    fusion_df.to_csv(raw_feature_output_folder, index=False)
    print(f"Batch {batch_count} saved: raw features to {output_file}")

def histogram_feature_save_batch(histogram_features, labels, batch_count):
    fusion_df = pd.DataFrame(histogram_features)
    fusion_df['label'] = labels
    output_file = os.path.join(histogram_feature_feature_output_folder, f"histogram_features_batch_{batch_count}.csv")
    fusion_df.to_csv(raw_feature_output_folder, index=False)
    print(f"Batch {batch_count} saved: histogram features to {output_file}")

def sequence_feature_save_batch(sequence_features, labels, batch_count):
    fusion_df = pd.DataFrame(sequence_features)
    fusion_df['label'] = labels
    output_file = os.path.join(sequence_feature_output_folder, f"sequence_features_batch_{batch_count}.csv")
    fusion_df.to_csv(raw_feature_output_folder, index=False)
    print(f"Batch {batch_count} saved: sequence features to {output_file}")

# 修改后的标签处理
def update_labels_with_preceding_faults(disk_data, look_back=10):
    """
    将标签为1的记录的前10条记录的标签也置为1
    :param disk_data: 输入的硬盘数据（包含标签列）
    :param look_back: 需要将故障标签传播到前面的记录数
    :return: 更新后的硬盘数据
    """
    labels = disk_data['label'].values  # 获取标签列

    # 遍历所有标签
    for i in range(look_back, len(labels)):  # 从look_back开始，避免索引越界
        if labels[i] == 1:
            start_idx = max(i - look_back, 0)  # 确保不越界
            labels[start_idx:i] = 1  # 将前10条记录的标签设置为1

    # 将更新后的标签回写到数据框
    disk_data['label'] = labels
    return disk_data


# 在提取特征之前，更新标签
def extract_and_save_all_features(data_folder, selected_features,selected_sequence_features, batch_size=50000):
    fusion_features_list = []
    histogram_features_all=[]
    sequence_features_list_all=[]
    raw_features_all=[]
    labels_list = []
    batch_count = 1
    disk_files = [f for f in os.listdir(data_folder) if f.startswith("disk_disk_")]
    disk_files.sort()  # 确保文件按字母顺序处理
    disk_files = disk_files[:10000]
    selected_features = [int(x) for x in selected_features]
    selected_sequence_features = [int(x) for x in selected_sequence_features]
    for disk_file in tqdm(disk_files, desc="Extracting and saving fusion features"):
        disk_data = pd.read_csv(os.path.join(data_folder, disk_file))

        # 更新标签，前10条记录如果标签为1，将标签置为1
        disk_data = update_labels_with_preceding_faults(disk_data, look_back=7)

        labels = disk_data['label'].values
        disk_data = disk_data.drop(columns=['label', 'serial_number', 'model', 'dt'], errors='ignore')

        # 计算全局最小值和最大值
        # min_vals, max_vals = calculate_min_max(data_folder)
        buckets = create_buckets(min_vals, max_vals, bucket_count)

        for start in range(0, len(disk_data) - 1, step_size):
            end = min(start + default_window_size, len(disk_data))
            window_data = disk_data.iloc[start:end].values
            if len(window_data) < 30:
                continue

                # 提取直方图特征（选择性特征）
            histogram_features = calculate_histogram_features(window_data, buckets)

            # Use list comprehension to select the features
            histogram_features = [histogram_features[idx] for idx in selected_features]

            # 提取原始特征
            raw_features = extract_raw_features(window_data)
            sequence_features_list=[]

            # 提取序列特征
            if len(window_data) == 0:  # 如果窗口为空，则跳过该窗口并插入NaN的特征
                sequence_features_list.append([np.nan] * (segments * 6))  # NaN placeholder for each segment's features
                labels_list.append(labels[start - 1])  # 保持标签的对齐
                continue

            # 计算序列特征

            for col in range(window_data.shape[1]):  # 遍历每个特征列
                feature_col = window_data[:, col]
                seq_features = calculate_sequence_features(feature_col, segments, mean_features)
                sequence_features_list.extend(seq_features)

            # Use list comprehension to select the features
            sequence_features_list = [sequence_features_list[idx] for idx in selected_sequence_features]
            # 合并特征
            histogram_features_all.append(histogram_features)
            sequence_features_list_all.append(sequence_features_list)
            raw_features_all.append(raw_features)

            raw_features = torch.tensor(raw_features) if not isinstance(raw_features, torch.Tensor) else raw_features
            histogram_features = torch.tensor(histogram_features) if not isinstance(histogram_features,
                                                                                    torch.Tensor) else histogram_features
            sequence_features_list = torch.tensor(sequence_features_list) if not isinstance(sequence_features_list,
                                                                                  torch.Tensor) else sequence_features_list
            # fusion_features = apply_attention_fusion(raw_features, histogram_features, sequence_features_list, 30)

            # 添加标签
            # fusion_features_list.append(fusion_features)
            labels_list.append(labels[end - 1])


            if len(raw_features_all) >= batch_size:
                # save_batch(fusion_features_list, labels_list, batch_count)
                histogram_feature_save_batch(histogram_features_all, labels_list, batch_count)
                sequence_feature_save_batch(sequence_features_list_all, labels_list, batch_count)
                raw_feature_save_batch(raw_features_all, labels_list, batch_count)
                # fusion_features_list.clear()
                histogram_features_all.clear()
                sequence_features_list_all.clear()
                raw_features_all.clear()
                labels_list.clear()
                batch_count += 1

    # 如果最后一批不满批次大小，也要保存
    if histogram_features_all:
        # save_batch(fusion_features_list, labels_list, batch_count)
        histogram_feature_save_batch(histogram_features_all, labels_list, batch_count)
        sequence_feature_save_batch(sequence_features_list_all, labels_list, batch_count)
        raw_feature_save_batch(raw_features_all, labels_list, batch_count)


# 调用主函数
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. 计算全局最小值和最大值
min_max_data = np.load(min_max_path, allow_pickle=True).item()
min_vals = min_max_data['min_vals']
max_vals = min_max_data['max_vals']
# 提取并保存所有特征

extract_and_save_all_features(data_folder, selected_histogram_features,selected_sequence_features,batch_size=batch_size)
#

