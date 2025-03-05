import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, raw_input_dim, hist_input_dim, output_dim):
        super().__init__()

        # 1. 多尺度卷积特征提取
        self.raw_convs = nn.ModuleList([
            nn.Conv1d(raw_input_dim, output_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]  # 不同尺度的卷积核
        ])

        self.hist_convs = nn.ModuleList([
            nn.Conv1d(hist_input_dim, output_dim, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7]
        ])

        # 2. 自适应特征权重
        self.raw_attention = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.Sigmoid()
        )

        self.hist_attention = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.Sigmoid()
        )

        # 3. 特征增强
        self.raw_enhance = nn.Sequential(
            nn.Linear(raw_input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )

        self.hist_enhance = nn.Sequential(
            nn.Linear(hist_input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )

        # 4. 交叉注意力
        self.cross_attn = nn.MultiheadAttention(output_dim, 4, batch_first=True)

        # 5. 最终融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, raw_features, hist_features):
        batch_size, seq_len, _ = raw_features.shape

        # 1. 提取多尺度卷积特征
        raw_conv_features = []
        hist_conv_features = []

        # 转换维度以适应卷积
        raw_input = raw_features.transpose(1, 2)  # [batch, channels, seq_len]
        hist_input = hist_features.transpose(1, 2)

        for conv in self.raw_convs:
            feat = conv(raw_input)
            raw_conv_features.append(feat.transpose(1, 2))  # 转回 [batch, seq_len, channels]

        for conv in self.hist_convs:
            feat = conv(hist_input)
            hist_conv_features.append(feat.transpose(1, 2))

        # 2. 合并多尺度特征
        raw_multi_scale = torch.cat(raw_conv_features, dim=-1)
        hist_multi_scale = torch.cat(hist_conv_features, dim=-1)

        # 3. 计算自适应权重
        raw_weights = self.raw_attention(raw_multi_scale)
        hist_weights = self.hist_attention(hist_multi_scale)

        # 4. 直接特征增强
        raw_direct = self.raw_enhance(raw_features.reshape(-1, raw_features.size(-1)))
        raw_direct = raw_direct.view(batch_size, seq_len, -1)

        hist_direct = self.hist_enhance(hist_features.reshape(-1, hist_features.size(-1)))
        hist_direct = hist_direct.view(batch_size, seq_len, -1)

        # 5. 加权融合
        raw_features = raw_direct * raw_weights
        hist_features = hist_direct * hist_weights

        # 6. 交叉注意力增强
        cross_raw, _ = self.cross_attn(raw_features, hist_features, hist_features)
        cross_hist, _ = self.cross_attn(hist_features, raw_features, raw_features)

        # 7. 最终特征融合
        final_features = torch.cat([
            raw_features,  # 原始特征
            hist_features,  # 直方图特征
            cross_raw,  # 交叉注意力特征
            cross_hist  # 交叉注意力特征
        ], dim=-1)

        output = self.fusion(final_features.reshape(-1, final_features.size(-1)))
        output = output.view(batch_size, seq_len, -1)

        return output


def process_merged_features(model, sampled_dir, batch_size=32):
    print("Loading merged features...")

    raw_features = np.load(os.path.join(sampled_dir, "merged_raw_features.npy"))
    hist_features = np.load(os.path.join(sampled_dir, "merged_hist_features.npy"))

    print(f"Raw features shape: {raw_features.shape}")
    print(f"Histogram features shape: {hist_features.shape}")

    def robust_scale(features):
        # 计算每个特征维度的统计量
        q75, q25 = np.percentile(features, [75, 25], axis=(0, 1))
        iqr = q75 - q25
        median = np.median(features, axis=(0, 1))

        # 处理异常值
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        features = np.clip(features, lower, upper)

        # 稳健缩放
        scaled = (features - median) / (iqr + 1e-8)
        return scaled.astype(np.float32)

    # 预处理特征
    raw_features = robust_scale(raw_features)
    hist_features = robust_scale(hist_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    num_samples = len(raw_features)
    all_outputs = []

    print("\nProcessing features in batches...")
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_end = min(i + batch_size, num_samples)

            batch_raw = torch.tensor(raw_features[i:batch_end], dtype=torch.float32).to(device)
            batch_hist = torch.tensor(hist_features[i:batch_end], dtype=torch.float32).to(device)

            batch_output = model(batch_raw, batch_hist).cpu().numpy()
            all_outputs.append(batch_output)

    fusion_output = np.concatenate(all_outputs, axis=0)

    # 后处理
    fusion_output = robust_scale(fusion_output)

    print(f"Fusion output shape: {fusion_output.shape}")

    save_path = os.path.join(sampled_dir, "merged_fusion_features.npy")
    print(f"\nSaving fusion features to: {save_path}")
    np.save(save_path, fusion_output)

    loaded_data = np.load(save_path)
    print(f"Verified saved data shape: {loaded_data.shape}")


def main():
    raw_input_dim = 39
    hist_input_dim = 288
    output_dim = 64

    print("Initializing model...")
    model = MultiScaleFeatureExtractor(raw_input_dim, hist_input_dim, output_dim)

    sampled_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence"

    if not os.path.exists(sampled_dir):
        print(f"Directory not found: {sampled_dir}")
        return

    process_merged_features(model, sampled_dir)
    print("\nFeature fusion complete!")


if __name__ == "__main__":
    main()