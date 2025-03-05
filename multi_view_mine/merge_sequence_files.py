import os
import numpy as np
from tqdm import tqdm


def normalize_features(data):
    """
    对3D特征进行归一化 (samples, features, timesteps)
    """
    # 计算每个时间步的全局最大最小值
    max_vals = data.max(axis=(0, 1))  # shape: (timesteps,)
    min_vals = data.min(axis=(0, 1))  # shape: (timesteps,)

    # 处理零除问题
    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1  # 避免除以零

    # 扩展维度以便广播
    max_vals = max_vals[None, None, :]  # shape: (1, 1, timesteps)
    min_vals = min_vals[None, None, :]  # shape: (1, 1, timesteps)
    denominator = denominator[None, None, :]  # shape: (1, 1, timesteps)

    # 归一化
    normalized_data = (data - min_vals) / denominator

    return normalized_data

def merge_all_files(sampled_dir):
    """合并所有特征文件和标签文件"""

    # 获取所有原始特征文件并排序
    raw_files = sorted([f for f in os.listdir(sampled_dir) if f.startswith("raw_feature_sequence_")])

    # 用于存储所有特征和标签
    all_raw_features = []
    all_hist_features = []
    all_emd_features = []  # 新增：EMD特征
    all_levels = []        # 新增：level数据
    all_labels = []

    print("Starting to merge files...")

    for raw_file in tqdm(raw_files, desc="Merging files"):
        # 获取文件索引
        file_index = raw_file.split("_")[-1].split(".")[0]

        # 构建文件路径
        raw_path = os.path.join(sampled_dir, f"raw_feature_sequence_{file_index}.npy")
        hist_path = os.path.join(sampled_dir, f"histogram_feature_sequence_{file_index}.npy")
        emd_path = os.path.join(sampled_dir, f"emd_feature_sequence_{file_index}.npy")  # 新增
        level_path = os.path.join(sampled_dir, f"level_{file_index}.npy")  # 新增
        label_path = os.path.join(sampled_dir, f"tag_{file_index}.npy")

        try:
            # 加载数据
            raw_data = np.load(raw_path, allow_pickle=True)
            hist_data = np.load(hist_path, allow_pickle=True)
            emd_data = np.load(emd_path, allow_pickle=True)  # 新增
            level_data = np.load(level_path, allow_pickle=True)  # 新增
            labels = np.load(label_path, allow_pickle=True)

            # 验证数据维度和对应关系
            assert len(raw_data) == len(hist_data) == len(emd_data) == len(level_data) == len(labels), \
                f"Sample count mismatch in file {file_index}: raw={len(raw_data)}, hist={len(hist_data)}, " \
                f"emd={len(emd_data)}, level={len(level_data)}, labels={len(labels)}"

            # 添加到列表中
            all_raw_features.append(raw_data)
            all_hist_features.append(hist_data)
            all_emd_features.append(emd_data)
            all_levels.append(level_data)
            all_labels.append(labels)

        except Exception as e:
            print(f"Error processing file {file_index}: {str(e)}")
            continue

    # 合并所有数据
    print("\nConcatenating all features and labels...")
    merged_raw = np.concatenate(all_raw_features, axis=0)
    merged_hist = np.concatenate(all_hist_features, axis=0)
    merged_emd = np.concatenate(all_emd_features, axis=0)  # 新增
    merged_levels = np.concatenate(all_levels, axis=0)     # 新增
    merged_labels = np.concatenate(all_labels, axis=0)

    # 验证最终数据
    assert len(merged_raw) == len(merged_hist) == len(merged_emd) == len(merged_levels) == len(merged_labels), \
        "Final sample count mismatch!"

    # 归一化
    merged_raw = normalize_features(merged_raw)
    merged_emd = normalize_features(merged_emd)  # 新增：对EMD特征也进行归一化

    # 保存合并后的数据
    print("\nSaving merged files...")
    np.save(os.path.join(sampled_dir, "merged_raw_features.npy"), merged_raw)
    np.save(os.path.join(sampled_dir, "merged_hist_features.npy"), merged_hist)
    np.save(os.path.join(sampled_dir, "merged_emd_features.npy"), merged_emd)  # 新增
    np.save(os.path.join(sampled_dir, "merged_levels.npy"), merged_levels)     # 新增
    np.save(os.path.join(sampled_dir, "merged_labels.npy"), merged_labels)

    # 打印统计信息
    print(f"\nMerging complete!")
    print(f"Total samples: {len(merged_raw)}")
    print(f"Raw features shape: {merged_raw.shape}")
    print(f"Histogram features shape: {merged_hist.shape}")
    print(f"EMD features shape: {merged_emd.shape}")      # 新增
    print(f"Level data shape: {merged_levels.shape}")     # 新增
    print(f"Labels shape: {merged_labels.shape}")
    print(f"Positive samples: {np.sum(merged_labels == 1)}")
    print(f"Negative samples: {np.sum(merged_labels == 0)}")


def main():
    sampled_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence"

    # 检查目录是否存在
    if not os.path.exists(sampled_dir):
        print(f"Directory not found: {sampled_dir}")
        return

    # 合并所有文件
    merge_all_files(sampled_dir)


if __name__ == "__main__":
    main()