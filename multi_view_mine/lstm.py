import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from tqdm import tqdm

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

# 数据目录
data_dir = "/home/wangy/code/EDITS复现/multi_view_mine/sampled_sequence_with_level_new"

def load_batch_data(data_dir, split='train'):
    """加载所有batch的数据"""
    batch_datasets = []
    
    batch_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{split}_hist_features_batch_")])
    batch_nums = set([f.split("_")[-1].split(".")[0] for f in batch_files])
    
    for batch_num in tqdm(batch_nums, desc=f"Loading {split} batches"):
        try:
            x = np.load(os.path.join(data_dir, f"{split}_hist_features_batch_{batch_num}.npy"))
            y = np.load(os.path.join(data_dir, f"{split}_labels_batch_{batch_num}.npy"))
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            
            batch_dataset = TensorDataset(x, y)
            batch_datasets.append(batch_dataset)
            
        except Exception as e:
            print(f"Error loading batch {batch_num}: {str(e)}")
            continue
    
    combined_dataset = ConcatDataset(batch_datasets)
    return combined_dataset

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

def main():
    # 加载训练集和测试集
    print("Loading datasets...")
    train_dataset = load_batch_data(data_dir, 'train')
    test_dataset = load_batch_data(data_dir, 'test')
    
    # 创建数据加载器
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        generator=g,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    # 初始化模型参数
    input_size = 288
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 100
    learning_rate = 1e-3

    # 实例化模型
    torch.manual_seed(SEED)
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    print("Starting training...")
    losses = []  # 记录训练损失

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 训练完成后进行评估
    print("\nTraining completed. Starting evaluation...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    predicted_labels = torch.cat(all_preds)
    true_labels = torch.cat(all_labels)

    # 计算评估指标
    accuracy = (predicted_labels == true_labels).float().mean().item()
    precision = precision_score(true_labels.numpy(), predicted_labels.numpy())
    recall = recall_score(true_labels.numpy(), predicted_labels.numpy())
    f1 = f1_score(true_labels.numpy(), predicted_labels.numpy())

    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"{data_dir}/final_hist_lstm_model.pth")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), losses, marker='.')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss")
    plt.grid(True)
    plt.savefig(f'{data_dir}/lstm_training_loss.png')
    plt.close()

if __name__ == "__main__":
    main()