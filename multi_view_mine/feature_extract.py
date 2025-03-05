import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ==================== ParallelEncoder: 处理单独特征 ====================
class ParallelEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(ParallelEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向 LSTM，所以是 2 倍 hidden_dim
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (seq_len, batch_size, hidden_dim*2)
        out = self.fc(lstm_out)  # 变成 (seq_len, batch_size, output_dim)
        out = self.layer_norm(out)
        out = torch.tanh(out)
        return out

# ==================== CrossEncoder: 交叉特征编码 ====================
class CrossEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers):
        super(CrossEncoder, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x1, x2):
        attn_output1, _ = self.attn(x1, x2, x2)
        attn_output2, _ = self.attn(x2, x1, x1)
        lstm_input1 = attn_output1 + x1  # 残差连接
        lstm_input2 = attn_output2 + x2
        lstm_out1, _ = self.lstm(lstm_input1)
        lstm_out2, _ = self.lstm(lstm_input2)
        out1 = self.fc(lstm_out1)
        out2 = self.fc(lstm_out2)
        out = (out1 + out2) / 2  # 融合两个分支
        out = self.layer_norm(out)
        out = torch.tanh(out)
        return out

# ==================== FeatureExtractor: 组合 Parallel 和 Cross Encoder ====================
class FeatureExtractor(nn.Module):
    def __init__(self, raw_input_dim, hist_input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super(FeatureExtractor, self).__init__()
        self.parallel_encoder_raw = ParallelEncoder(raw_input_dim, hidden_dim, output_dim, num_layers)
        self.parallel_encoder_hist = ParallelEncoder(hist_input_dim, hidden_dim, output_dim, num_layers)
        self.cross_encoder = CrossEncoder(output_dim, num_heads, hidden_dim, num_layers)
        self.fc_fusion = nn.Linear(output_dim * 3, output_dim)
        self.layer_norm_fusion = nn.LayerNorm(output_dim)

    def forward(self, raw_features, histogram_features):
        raw_features = raw_features.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        histogram_features = histogram_features.permute(1, 0, 2)

        # 经过 ParallelEncoder 变换维度
        parallel_out_raw = self.parallel_encoder_raw(raw_features)  # (seq_len, batch_size, 64)
        parallel_out_hist = self.parallel_encoder_hist(histogram_features)  # (seq_len, batch_size, 64)

        # 交叉编码
        cross_out = self.cross_encoder(parallel_out_raw, parallel_out_hist)  # (seq_len, batch_size, 64)

        # 在序列维度上进行特征拼接
        o_parallel = torch.cat((parallel_out_raw, parallel_out_hist), dim=-1)  # (seq_len, batch_size, 128)
        o_parallel = torch.tanh(o_parallel)

        # 融合所有特征
        o_fusion = torch.cat((o_parallel, cross_out), dim=-1)  # (seq_len, batch_size, 192)
        o_fusion = self.fc_fusion(o_fusion)  # (seq_len, batch_size, 64)
        o_fusion = self.layer_norm_fusion(o_fusion)
        o_fusion = torch.tanh(o_fusion)

        # 将维度顺序改回 batch_first
        o_fusion = o_fusion.permute(1, 0, 2)  # (batch_size, seq_len, 64)

        return o_fusion

# ==================== 处理数据并保存 ====================
import os
import numpy as np
import torch

def process_and_save_feature(model, raw_file, histogram_file, save_path):
    # Load arrays
    raw_features = np.load(raw_file, allow_pickle=True)
    histogram_features = np.load(histogram_file, allow_pickle=True)

    # Convert object arrays to float arrays
    raw_features = np.array(raw_features.tolist(), dtype=np.float32)
    histogram_features = np.array(histogram_features.tolist(), dtype=np.float32)

    # 获取序列长度
    sequence_length = raw_features.shape[1] if len(raw_features.shape) > 2 else 1

    # Process in batches
    batch_size = 32
    num_samples = len(raw_features)
    all_outputs = []

    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_raw = torch.tensor(raw_features[i:batch_end], dtype=torch.float32)
        batch_hist = torch.tensor(histogram_features[i:batch_end], dtype=torch.float32)

        with torch.no_grad():
            # 获取模型输出
            batch_output = model(batch_raw, batch_hist).detach().cpu().numpy()
            # 确保维度正确
            if len(batch_output.shape) == 3:
                all_outputs.append(batch_output)
            else:
                print("Warning: Unexpected output shape:", batch_output.shape)

    # 沿批次维度连接
    fusion_output = np.concatenate(all_outputs, axis=0)

    # 打印shape以检查
    print(f"Raw features shape: {raw_features.shape}")
    print(f"Model output shape before save: {fusion_output.shape}")

    # 提取文件编号
    raw_filename = os.path.basename(raw_file)  # 获取文件名
    file_index = raw_filename.split("_")[-1].split(".")[0]  # 提取编号部分

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 生成保存文件名
    save_file = os.path.join(save_path, f"fusion_sequence_{file_index}.npy")

    # 保存数据
    np.save(save_file, fusion_output)

    # 验证保存的数据
    loaded_data = np.load(save_file)
    print(f"Loaded data shape: {loaded_data.shape}")


def process_all_data(model, sequence_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    raw_files = sorted([f for f in os.listdir(sequence_path) if 'raw' in f])
    histogram_files = sorted([f for f in os.listdir(sequence_path) if 'histogram' in f])

    for raw_file, histogram_file in zip(raw_files, histogram_files):
        raw_file_path = os.path.join(sequence_path, raw_file)
        histogram_file_path = os.path.join(sequence_path, histogram_file)
        process_and_save_feature(model, raw_file_path, histogram_file_path, save_path)

# ==================== 主程序 ====================
if __name__ == "__main__":
    raw_input_dim = 39
    hist_input_dim = 288
    hidden_dim = 128
    output_dim = 64
    num_layers = 2
    num_heads = 4

    model = FeatureExtractor(raw_input_dim, hist_input_dim, hidden_dim, output_dim, num_layers, num_heads)

    sequence_path = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence"
    save_path = "/home/wangy/code/EDITS复现/multi_view_mine/fusion_features"

    # 检查输入数据维度
    sample_raw = np.load(os.path.join(sequence_path, "raw_feature_sequence_1.npy"),allow_pickle=True)
    print(f"Sample input shape: {sample_raw.shape}")

    # 如果输入数据是二维的，需要添加序列维度
    if len(sample_raw.shape) == 2:
        print("Warning: Input data is 2D, should be 3D for sequence processing")
        # 可以在这里决定如何处理二维数据
        # 例如：将数据重组成固定长度的序列

    process_all_data(model, sequence_path, save_path)
