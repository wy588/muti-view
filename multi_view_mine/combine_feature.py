import os
import numpy as np
from tqdm import tqdm


def combine_features(sampled_dir, split='train'):
    """合并指定split的原始特征、直方图特征和EMD特征，并保存不同组合"""
    print(f"Loading {split} features...")

    # 获取所有batch文件
    batch_files = sorted([f for f in os.listdir(sampled_dir) if f.startswith(f"{split}_raw_features_batch_")])
    batch_nums = set([f.split("_")[-1].split(".")[0] for f in batch_files])
    
    for batch_num in tqdm(batch_nums, desc=f"Processing {split} batches"):
        try:
            # 构建文件路径
            raw_path = os.path.join(sampled_dir, f"{split}_raw_features_batch_{batch_num}.npy")
            hist_path = os.path.join(sampled_dir, f"{split}_hist_features_batch_{batch_num}.npy")
            emd_path = os.path.join(sampled_dir, f"{split}_emd_features_batch_{batch_num}.npy")

            # 加载特征
            raw_data = np.load(raw_path, allow_pickle=True)
            hist_data = np.load(hist_path, allow_pickle=True)
            emd_data = np.load(emd_path, allow_pickle=True)

            # 验证维度
            assert raw_data.shape[1:] == (30, 39), f"Raw feature shape error: {raw_data.shape}"
            assert hist_data.shape[1:] == (30, 288), f"Histogram feature shape error: {hist_data.shape}"
            assert emd_data.shape[1:] == (30, 39), f"EMD feature shape error: {emd_data.shape}"
            assert len(raw_data) == len(hist_data) == len(emd_data), \
                f"Sample count mismatch: raw={len(raw_data)}, hist={len(hist_data)}, emd={len(emd_data)}"

            print(f"\nCombining features for batch {batch_num}...")
            # 原始特征 + EMD特征
            raw_emd_combined = np.concatenate([raw_data, emd_data], axis=2)  # (samples, 30, 39+39)
            raw_emd_path = os.path.join(sampled_dir, f"{split}_raw_emd_features_batch_{batch_num}.npy")
            np.save(raw_emd_path, raw_emd_combined)
            print(f"Raw + EMD features shape: {raw_emd_combined.shape}")

            # 直方图特征 + EMD特征
            hist_emd_combined = np.concatenate([hist_data, emd_data], axis=2)  # (samples, 30, 288+39)
            hist_emd_path = os.path.join(sampled_dir, f"{split}_hist_emd_features_batch_{batch_num}.npy")
            np.save(hist_emd_path, hist_emd_combined)
            print(f"Histogram + EMD features shape: {hist_emd_combined.shape}")

            # 原始特征 + 直方图特征
            raw_hist_combined = np.concatenate([raw_data, hist_data], axis=2)  # (samples, 30, 39+288)
            raw_hist_path = os.path.join(sampled_dir, f"{split}_raw_hist_features_batch_{batch_num}.npy")
            np.save(raw_hist_path, raw_hist_combined)
            print(f"Raw + Histogram features shape: {raw_hist_combined.shape}")

            # 所有特征组合
            combined_data = np.concatenate([raw_data, hist_data, emd_data], axis=2)
            combined_path = os.path.join(sampled_dir, f"{split}_combined_features_batch_{batch_num}.npy")
            np.save(combined_path, combined_data)
            print(f"All combined features shape: {combined_data.shape}")

        except Exception as e:
            print(f"Error processing batch {batch_num}: {str(e)}")
            continue


def main():
    sampled_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence_with_level_new"

    # 检查目录是否存在
    if not os.path.exists(sampled_dir):
        print(f"Directory not found: {sampled_dir}")
        return

    # 合并训练集特征
    combine_features(sampled_dir, 'train')
    
    # 合并测试集特征
    combine_features(sampled_dir, 'test')

    print("\nAll feature combinations saved successfully!")


if __name__ == "__main__":
    main()