import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

# 参数设置
data_folder = "/home/wangy/code/EDITS复现/multi-view/sorted_disk_data"
default_window_size = 256
step_size = 1
# 创建EMD特征保存目录
output_folder = "/home/wangy/code/EDITS复现/multi_view_mine/emd_feature"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
def extract_and_save_emd_features(data_folder, start_index=33000):

    disk_files = [f for f in os.listdir(data_folder) if f.startswith("disk_disk_")]
    disk_files.sort()

    # 从指定索引开始处理
    disk_files = disk_files[start_index:]
    print(f"Continuing from index {start_index}, first file to process: {disk_files[0]}")


    for disk_file in tqdm(disk_files, desc="Extracting and saving EMD features"):
        # 读取数据
        disk_data = pd.read_csv(os.path.join(data_folder, disk_file))
        original_labels = disk_data['label'].values
        disk_data = disk_data.drop(columns=['label', 'serial_number', 'model', 'dt'], errors='ignore')

        emd_features_list = []
        labels_list = []


        # 对每个时间点生成特征
        for i in range(len(disk_data)):
            start = max(0, i - default_window_size + 1)
            end = i + 1
            window_data = disk_data.iloc[start:end].values

            # 计算EMD特征
            emd_features = calculate_emd_distance(window_data)
            emd_features_list.append(emd_features)
            labels_list.append(original_labels[i])

        # 验证标签对应关系
        if len(emd_features_list) != len(disk_data):
            print(
                f"Warning: EMD feature count ({len(emd_features_list)}) doesn't match original data length ({len(disk_data)})")

        # 保存特征
        save_emd_features(emd_features_list, labels_list, disk_file.split('.')[0])

        # 清理内存
        del emd_features_list
        del labels_list
        gc.collect()


def save_emd_features(emd_features, labels, disk_file_name):
    """
    保存EMD特征到文件
    """
    if len(emd_features) != len(labels):
        print(f"Error: Length mismatch in save_emd_features - Features: {len(emd_features)}, Labels: {len(labels)}")
        return

    try:
        emd_df = pd.DataFrame(emd_features)
        emd_df['label'] = labels
        output_file = os.path.join(output_folder, f"emd_features_{disk_file_name}.csv")
        emd_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error saving EMD features: {e}")
    finally:
        del emd_df  # 释放内存


def calculate_emd_distance(window_data):
    """
    计算窗口内两半数据之间的推土机距离
    :param window_data: 窗口数据，shape为(window_size, feature_dim)
    :return: 每个特征维度的EMD距离
    """
    if len(window_data) < 2:
        return np.zeros(window_data.shape[1])

    # 将窗口分成两半
    mid_point = len(window_data) // 2
    first_half = window_data[:mid_point]
    second_half = window_data[mid_point:]

    # 初始化EMD距离数组
    emd_distances = np.zeros(window_data.shape[1])

    # 对每个特征维度计算EMD
    for i in range(window_data.shape[1]):
        # 获取该维度的数据
        dist1 = first_half[:, i]
        dist2 = second_half[:, i]

        # 计算累积分布函数（CDF）
        sorted_dist1 = np.sort(dist1)
        sorted_dist2 = np.sort(dist2)

        # 生成均匀分布的点
        n1, n2 = len(dist1), len(dist2)
        positions1 = np.linspace(0, 1, n1)
        positions2 = np.linspace(0, 1, n2)

        # 处理不同长度的分布
        if n1 != n2:
            if n1 < n2:
                sorted_dist1 = np.interp(positions2, positions1, sorted_dist1)
            else:
                sorted_dist2 = np.interp(positions1, positions2, sorted_dist2)

        # 计算EMD
        emd_distances[i] = np.abs(sorted_dist1 - sorted_dist2).mean()

    return emd_distances


# 主程序调用
if __name__ == "__main__":

    # 提取并保存EMD特征
    extract_and_save_emd_features(data_folder,start_index=33000)