import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib
# 设置 Agg 后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_tensor
from torch.utils.tensorboard import SummaryWriter


# 读取数据
file_path = './Summary of Weather.csv'
data = pd.read_csv(file_path)

# 提取最高气温列
max_temp = data['MaxTemp'].values.reshape(-1, 1)

# 准备数据集类
class WeatherDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length - 5 + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        # 同时获取未来 1 天和未来 5 天的真实值
        y_1 = self.data[idx + self.seq_length:idx + self.seq_length + 1]
        y_5 = self.data[idx + self.seq_length:idx + self.seq_length + 5]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_1, dtype=torch.float32), torch.tensor(y_5, dtype=torch.float32)

seq_length = 30
train_size = int(len(max_temp) * 0.8)

# 数据集
train_dataset = WeatherDataset(max_temp[:train_size], seq_length)
test_dataset = WeatherDataset(max_temp[train_size:], seq_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义 RNN 模型
class WeatherRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 输出 6 个值，第一个为未来 1 天预测，后 5 个为未来 5 天预测
        self.fc = nn.Linear(hidden_size, 6)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型、损失函数和优化器
model = WeatherRNNModel(input_size=1, hidden_size=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 初始化 TensorBoard
writer = SummaryWriter(os.path.join('runs', 'weather_prediction'))

# 初始化用于存储损失的列表
train_losses = []
test_losses = []

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for inputs, labels_1, labels_5 in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        output_1 = outputs[:, 0].unsqueeze(-1)
        output_5 = outputs[:, 1:]
        loss_1 = criterion(output_1, labels_1.squeeze(-1))
        loss_5 = criterion(output_5, labels_5.squeeze(-1))
        loss = loss_1 + loss_5
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)  # 存储训练损失

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, labels_1, labels_5 in test_loader:
            outputs = model(inputs)
            output_1 = outputs[:, 0].unsqueeze(-1)
            output_5 = outputs[:, 1:]
            loss_1 = criterion(output_1, labels_1.squeeze(-1))
            loss_5 = criterion(output_5, labels_5.squeeze(-1))
            loss = loss_1 + loss_5
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)  # 存储测试损失

    # 记录损失到 TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
# 关闭 TensorBoard writer
writer.close()

# 保存模型
torch.save(model.state_dict(), 'weather_model.pth')

# 重新创建模型实例
model = WeatherRNNModel(input_size=1, hidden_size=50)
# 加载模型状态字典
model.load_state_dict(torch.load('weather_model.pth'))

# 收集预测值和真实值
all_preds_1 = []
all_labels_1 = []
all_preds_5 = []
all_labels_5 = []

model.eval()
with torch.no_grad():
    for inputs, labels_1, labels_5 in test_loader:
        outputs = model(inputs)
        output_1 = outputs[:, 0].unsqueeze(-1)
        output_5 = outputs[:, 1:]
        all_preds_1.extend(output_1.squeeze().tolist())
        all_labels_1.extend(labels_1.squeeze().tolist())
        all_preds_5.extend(output_5.squeeze().tolist())
        all_labels_5.extend(labels_5.squeeze().tolist())

# 取前10日真实值
last_10_days_labels = all_labels_1[-10:]

# 获取未来1日和5日预测值
last_prediction_1_day = all_preds_1[-1]
last_prediction_5_days = all_preds_5[-1] if isinstance(all_preds_5[-1], list) else [all_preds_5[-1]]

# 组合数据
combined_1_day = last_10_days_labels + [last_prediction_1_day]
combined_5_days = last_10_days_labels + last_prediction_5_days

# 重新初始化 TensorBoard writer 或者使用之前的
writer = SummaryWriter(os.path.join('runs', 'weather_prediction'))


# 绘制 1 天预测图像
plt.figure(figsize=(12, 6))
x_1 = range(1, len(combined_1_day) + 1)
plt.plot(x_1, combined_1_day, label='Predicted (1 day)')
plt.axvline(x=10, color='r', linestyle='--', label='Prediction Start')
plt.title('1-Day Weather Prediction')
plt.xlabel('Day')
plt.ylabel('Max Temperature')
# 手动设置 x 轴刻度
plt.xticks(x_1)
plt.legend()

# 将图像保存到缓冲区
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = to_tensor(plt.imread(buf))
writer.add_image('1-Day Prediction', image)
plt.close()

# 绘制 5 天预测图像
plt.figure(figsize=(12, 6))
x_5 = range(1, len(combined_5_days) + 1)
plt.plot(x_5, combined_5_days, label='Predicted (5 days)')
plt.axvline(x=10, color='r', linestyle='--', label='Prediction Start')
plt.title('5-Day Weather Prediction')
plt.xlabel('Days')
plt.ylabel('Max Temperature')
# 手动设置 x 轴刻度
plt.xticks(x_5)
plt.legend()

# 将图像保存到缓冲区
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = to_tensor(plt.imread(buf))
writer.add_image('5-Day Prediction', image)
plt.close()

writer.close()


# 预测未来
last_sequence = torch.tensor(max_temp[-seq_length:], dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    output = model(last_sequence)
    prediction_1_day = output[0, 0].item()
    prediction_5_days = output[0, 1:].tolist()

    print(f"未来 1 天的最高气温预测值: {prediction_1_day:.2f}")
    print("未来连续 5 天的最高气温预测值:")
    for i, temp in enumerate(prediction_5_days, start=1):
        print(f"第 {i} 天: {temp:.2f}")