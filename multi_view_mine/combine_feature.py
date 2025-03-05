import os
import numpy as np


def combine_merged_features(sampled_dir):
    """合并已经合并好的原始特征、直方图特征和EMD特征"""
    print("Loading merged features...")

    # 构建文件路径
    raw_path = os.path.join(sampled_dir, "merged_raw_features.npy")
    hist_path = os.path.join(sampled_dir, "merged_hist_features.npy")
    emd_path = os.path.join(sampled_dir, "merged_emd_features.npy")  # 新增

    try:
        # 加载特征
        raw_data = np.load(raw_path, allow_pickle=True)
        hist_data = np.load(hist_path, allow_pickle=True)
        emd_data = np.load(emd_path, allow_pickle=True)   # 新增

        # 验证维度
        assert raw_data.shape[1:] == (30, 39), f"Raw feature shape error: {raw_data.shape}"
        assert hist_data.shape[1:] == (30, 288), f"Histogram feature shape error: {hist_data.shape}"
        assert emd_data.shape[1:] == (30, 39), f"EMD feature shape error: {emd_data.shape}"  # 新增
        assert len(raw_data) == len(hist_data) == len(emd_data), \
            f"Sample count mismatch: raw={len(raw_data)}, hist={len(hist_data)}, emd={len(emd_data)}"

        print("\nCombining features...")
        # 在第三个维度（特征维度）上拼接所有特征
        combined_data = np.concatenate([raw_data, hist_data, emd_data], axis=2)  # (samples, 30, 39+288+39)

        # 保存合并后的特征
        combined_path = os.path.join(sampled_dir, "merged_combined_features_new.npy")
        print("\nSaving combined features...")
        np.save(combined_path, combined_data)

        print(f"\nFeatures combined successfully!")
        print(f"Combined features shape: {combined_data.shape}")
        print(f"Saved to: {combined_path}")

    except Exception as e:
        print(f"Error combining features: {str(e)}")


def main():
    sampled_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence"

    # 检查目录是否存在
    if not os.path.exists(sampled_dir):
        print(f"Directory not found: {sampled_dir}")
        return

    # 合并特征
    combine_merged_features(sampled_dir)


if __name__ == "__main__":
    main()