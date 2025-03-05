import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

# 设定随机种子，保证完全复现性
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
feature_path = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence/merged_combined_features.npy"
label_path = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence/merged_labels.npy"

# 加载并转换数据类型
x_all = np.load(feature_path, allow_pickle=True)
y_all = np.load(label_path, allow_pickle=True)

# 转换数据类型
x_all = np.array([np.array(x, dtype=np.float32) for x in x_all])
y_all = np.array(y_all, dtype=np.float32)

print(f"Data shapes - Features: {x_all.shape}, Labels: {y_all.shape}")
print(f"Data types - Features: {x_all.dtype}, Labels: {y_all.dtype}")

# 转换为 PyTorch 张量
x_all = torch.tensor(x_all, dtype=torch.float32)
y_all = torch.tensor(y_all, dtype=torch.float32)

# 使用固定的随机索引进行划分
indices = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(SEED))
train_size = int(0.8 * len(x_all))

# 划分训练集和测试集
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# 使用索引获取数据
x_train = x_all[train_indices]
y_train = y_all[train_indices]
x_test = x_all[test_indices]
y_test = y_all[test_indices]


# 创建数据加载器，设置随机种子
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=128,
    shuffle=True,
    generator=g,
    worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
)

test_loader = DataLoader(
    TensorDataset(x_test, y_test),
    batch_size=64,
    shuffle=False
)


# LSTM 模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型参数
input_size = 327
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 1e-3

# 实例化模型，确保权重初始化的随机性也是固定的
torch.manual_seed(SEED)
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
f1_scores = []
best_f1 = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0

    for i, (x_batch, y_batch) in enumerate(train_loader):
        if i == 0 and epoch == 1:
            print("First batch first sample:", x_batch[0, 0, :5].tolist())  # 验证随机性

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 每 100 轮评估一次
    if epoch % 100 == 0:
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())

        predicted_labels = torch.cat(all_preds)
        true_labels = torch.cat(all_labels)

        accuracy = (predicted_labels == true_labels).float().mean().item()
        precision = precision_score(true_labels.numpy(), predicted_labels.numpy())
        recall = recall_score(true_labels.numpy(), predicted_labels.numpy())
        f1 = f1_score(true_labels.numpy(), predicted_labels.numpy())
        f1_scores.append(f1)

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"/home/wangy/code/EDITS复现/merged_sample_lstm_model_combined_best.pth")

        print(f"Epoch {epoch}:")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# 绘制 F1 分数趋势图
# plt.figure(figsize=(10, 6))
# plt.plot(range(100, num_epochs + 1, 100), f1_scores, marker='o')
# plt.xlabel("Epoch")
# plt.ylabel("F1 Score")
# plt.title("LSTM Training F1 Score")
# plt.grid(True)
# plt.savefig('/home/wangy/code/EDITS复现/lstm_hist_f1_score.png')
# plt.show()