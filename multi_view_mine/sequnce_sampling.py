import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def find_optimal_k(data, k_range):
    """使用轮廓系数(Silhouette Score)和肘部法则结合来找到最佳的聚类数"""
    inertias = []
    silhouette_scores = []
    k_values = range(2, min(k_range + 1, len(data)))

    # 计算不同k值的轮廓系数和惯性
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # 计算惯性的变化率
    inertia_changes = np.diff(inertias) / np.array(inertias[:-1])

    # 调整阈值，使其更容易选择较小的k值
    threshold = 0.15  # 增大阈值
    elbow_k = None
    for i, change in enumerate(inertia_changes):
        if abs(change) < threshold:
            elbow_k = i + 2
            break

    # 如果没有找到明显的肘部点，使用最大轮廓系数对应的k值
    if elbow_k is None:
        elbow_k = k_values[np.argmax(silhouette_scores)]

    # 缩小搜索窗口，更倾向于选择较小的k值
    window = 1  # 减小窗口大小
    start_k = max(2, elbow_k - window)
    end_k = min(len(k_values), elbow_k + window + 1)
    best_k = k_values[start_k - 2:end_k - 2][np.argmax(silhouette_scores[start_k - 2:end_k - 2])]

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow point (k={elbow_k})')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'go-')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return best_k


def process_file(raw_path, hist_path, label_path, emd_path, level_path, sampled_dir, file_index, first_file=False):
    """处理单个文件的聚类和筛选，并保存到新目录"""
    print(f"\nProcessing file {file_index}")

    # 创建保存采样数据的子目录
    os.makedirs(sampled_dir, exist_ok=True)

    # 构建采样后的文件路径
    sampled_raw_path = os.path.join(sampled_dir, f"raw_feature_sequence_{file_index}.npy")
    sampled_hist_path = os.path.join(sampled_dir, f"histogram_feature_sequence_{file_index}.npy")
    sampled_emd_path = os.path.join(sampled_dir, f"emd_feature_sequence_{file_index}.npy")
    sampled_level_path = os.path.join(sampled_dir, f"level_{file_index}.npy")
    sampled_label_path = os.path.join(sampled_dir, f"tag_{file_index}.npy")


    raw_data = np.load(raw_path, allow_pickle=True)
    hist_data = np.load(hist_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    emd_data = np.load(emd_path, allow_pickle=True)
    level_data = np.load(level_path, allow_pickle=True)

    # 确保数据是float32类型
    raw_data = np.array([np.array(x, dtype=np.float32) for x in raw_data])
    hist_data = np.array([np.array(x, dtype=np.float32) for x in hist_data])
    labels = np.array(labels, dtype=np.float32)
    level_data = np.array(level_data, dtype=np.float32)

    # 获取负样本索引
    negative_indices = np.where(labels == 0)[0]
    negative_data = raw_data[negative_indices]  # 使用原始特征序列

    # 将3D数据展平为2D进行聚类
    negative_data_reshaped = negative_data.reshape(negative_data.shape[0], -1)

    # 找到最佳聚类数
    optimal_k = find_optimal_k(negative_data_reshaped, k_range=10)
    print(f"Optimal number of clusters: {optimal_k}")

    # 执行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(negative_data_reshaped)

    # 计算每个样本到其聚类中心的距离
    distances = np.zeros(len(negative_indices))
    for i in range(len(negative_indices)):
        distances[i] = np.linalg.norm(negative_data_reshaped[i] - kmeans.cluster_centers_[cluster_labels[i]])

    # 对每个聚类选择距离最近的6%的样本
    selected_negative_indices = []
    for cluster in range(optimal_k):
        cluster_mask = cluster_labels == cluster
        cluster_distances = distances[cluster_mask]
        cluster_indices = negative_indices[cluster_mask]

        # 计算要保留的样本数量（6%）
        n_select = max(1, int(0.06 * len(cluster_indices)))

        # 选择距离最近的样本
        closest_indices = cluster_indices[np.argsort(cluster_distances)[:n_select]]
        selected_negative_indices.extend(closest_indices)

    # 在 process_file 函数中的绘图部分
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(negative_data_reshaped)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                          c=cluster_labels, cmap='viridis', s=5, alpha=0.5)




    plt.title(f'Clustered Negative Samples (File {file_index})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # 获取正样本索引
    positive_indices = np.where(labels == 1)[0]

    # 合并正样本索引和选定的负样本索引
    selected_indices = np.concatenate([positive_indices, selected_negative_indices])
    selected_indices.sort()

    # 保存筛选后的数据到新目录
    np.save(sampled_raw_path, raw_data[selected_indices])
    np.save(sampled_hist_path, hist_data[selected_indices])
    np.save(sampled_emd_path, emd_data[selected_indices])
    np.save(sampled_level_path, level_data[selected_indices])
    np.save(sampled_label_path, labels[selected_indices])

    print(f"File {file_index} processed. Original samples: {len(labels)}, "
          f"Selected samples: {len(selected_indices)}, "
          f"Positive samples: {len(positive_indices)}, "
          f"Selected negative samples: {len(selected_negative_indices)}")

    print(f"Sampled data saved to: {sampled_dir}")

    return len(labels), len(selected_indices)

def main():
    sequence_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sequence_feature_with_level"
    sampled_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence_with_level"

    # 创建采样数据保存目录
    os.makedirs(sampled_dir, exist_ok=True)

    # 获取所有原始特征序列文件
    raw_files = sorted([f for f in os.listdir(sequence_dir) if f.startswith("raw_feature_sequence_")])

    total_stats = []
    for i, raw_file in enumerate(tqdm(raw_files, desc="Processing files")):
        file_index = raw_file.split("_")[-1].split(".")[0]

        # 构建所有相关文件的路径
        raw_path = os.path.join(sequence_dir, f"raw_feature_sequence_{file_index}.npy")
        hist_path = os.path.join(sequence_dir, f"histogram_feature_sequence_{file_index}.npy")
        emd_path = os.path.join(sequence_dir, f"emd_feature_sequence_{file_index}.npy")
        level_path = os.path.join(sequence_dir, f"level_{file_index}.npy")
        label_path = os.path.join(sequence_dir, f"tag_{file_index}.npy")

        # 检查所有文件是否存在
        if not all(os.path.exists(p) for p in [raw_path, hist_path, label_path, emd_path, level_path]):
            print(f"Skipping file {file_index} due to missing files")
            continue

        try:
            original_count, selected_count = process_file(
                raw_path, hist_path, label_path, emd_path, level_path,
                sampled_dir, file_index, first_file=(i == 0)
            )
            total_stats.append((original_count, selected_count))
        except Exception as e:
            print(f"Error processing file {file_index}: {str(e)}")
            continue

    # 打印总体统计信息
    total_original = sum(orig for orig, _ in total_stats)
    total_selected = sum(sel for _, sel in total_stats)
    print(f"\nTotal processing complete:")
    print(f"Total original samples: {total_original}")
    print(f"Total selected samples: {total_selected}")
    print(f"Reduction ratio: {(total_original - total_selected) / total_original * 100:.2f}%")
    print(f"All sampled data saved to: {sampled_dir}")


if __name__ == "__main__":
    main()
