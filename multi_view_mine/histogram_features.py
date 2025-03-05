import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# 参数设置
data_folder = "/home/wangy/code/EDITS复现/multi-view/sorted_disk_data"
output_folder = "/home/wangy/code/EDITS复现/multi_view_mine/histogram_feature"
default_window_size = 256
step_size = 1
bucket_count = 100  # M，桶的数量
min_max_path = "/home/wangy/code/EDITS复现/multi-view/min_max_values.npy"
batch_size = 50000  # 每批次存储的记录数
mean_features_file = "/home/wangy/code/EDITS复现/multi-view_mine/mean_sequence_features.csv"
selected_features_path = "/home/wangy/code/EDITS复现/multi-view/selected_histogram_features_bucket/selected_features_union_new.npy"
# 加载选中的直方图特征
selected_histogram_features = np.load(selected_features_path, allow_pickle=True)


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



# 在提取特征之前，更新标签
def extract_and_save_all_features(data_folder, selected_features, start_index=130500):
    """
    从排序后的文件列表的指定索引继续处理
    :param data_folder: 数据文件夹路径
    :param selected_features: 选定的特征
    :param start_index: 从文件列表的第几个开始处理
    """
    disk_files = [f for f in os.listdir(data_folder) if f.startswith("disk_disk_")]
    disk_files.sort()

    # 从指定索引开始处理
    disk_files = disk_files[start_index:]
    print(f"Continuing from index {start_index}, first file to process: {disk_files[0]}")

    selected_features = [int(x) for x in selected_features]

    for disk_file in tqdm(disk_files, desc="Extracting and saving fusion features"):
        # 读取数据
        disk_data = pd.read_csv(os.path.join(data_folder, disk_file))
        original_labels = disk_data['label'].values
        disk_data = disk_data.drop(columns=['label', 'serial_number', 'model', 'dt'], errors='ignore')

        histogram_features_list = []
        labels_list = []

        # 计算全局最小值和最大值
        min_max_data = np.load(min_max_path, allow_pickle=True).item()
        min_vals = min_max_data['min_vals']
        max_vals = min_max_data['max_vals']
        buckets = create_buckets(min_vals, max_vals, bucket_count)

        # 对每个时间点生成直方图特征
        for i in range(len(disk_data)):
            start = max(0, i - default_window_size + 1)
            end = i + 1
            window_data = disk_data.iloc[start:end].values
            histogram_features = calculate_histogram_features(window_data, buckets)
            histogram_features = [histogram_features[idx] for idx in selected_features]
            histogram_features_list.append(histogram_features)
            labels_list.append(original_labels[i])

        # 验证标签对应关系
        if len(histogram_features_list) != len(disk_data):
            print(
                f"Warning: Feature count ({len(histogram_features_list)}) doesn't match original data length ({len(disk_data)})")

        # 保存特征
        save_batch(histogram_features_list, labels_list, disk_file.split('.')[0], disk_file.split('.')[0])

        # 清理内存
        del histogram_features_list
        del labels_list
        gc.collect()


def save_batch(histogram_features, labels, batch_count, disk_file_name):
    if len(histogram_features) != len(labels):
        print(f"Error: Length mismatch in save_batch - Features: {len(histogram_features)}, Labels: {len(labels)}")
        return

    try:
        histogram_df = pd.DataFrame(histogram_features)
        histogram_df['label'] = labels
        output_file = os.path.join(output_folder, f"histogram_features_{disk_file_name}.csv")
        histogram_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error saving batch: {e}")
    finally:
        del histogram_df  # 释放内存


# 调用主函数
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 提取并保存所有特征
extract_and_save_all_features(data_folder, selected_histogram_features, start_index=130500)
