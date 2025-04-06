import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter



# 加载Olivetti人脸数据集
faces = fetch_olivetti_faces(data_home='./face_data', shuffle=True)
X = faces.images
y = faces.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 将图像数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# 定义不同的RNN结构
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=4096,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128, 40)  # 40个类别

    def forward(self, x):
        x = x.reshape(x.size(0), 1, -1)  # 调整形状以适应RNN输入
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=4096,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128, 40)  # 40个类别

    def forward(self, x):
        x = x.reshape(x.size(0), 1, -1)  # 调整形状以适应LSTM输入
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=4096,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128, 40)  # 40个类别

    def forward(self, x):
        x = x.reshape(x.size(0), 1, -1)  # 调整形状以适应GRU输入
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class BiRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.birnn = nn.RNN(
            input_size=4096,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # 设置为双向
        )
        self.fc = nn.Linear(128 * 2, 40)  # 双向需要乘以2

    def forward(self, x):
        x = x.reshape(x.size(0), 1, -1)  # 调整形状以适应RNN输入
        out, _ = self.birnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 训练和评估函数
def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, num_epochs=300, lr=0.001):
    writer = SummaryWriter(f'runs/{model_name}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # 记录训练损失
        writer.add_scalar('Training Loss', loss.item(), epoch)

        #计算训练准确率
        # 计算训练准确率
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            # 记录训练准确率
            writer.add_scalar('Training Accuracy', train_accuracy, epoch)

        if (epoch + 1) % 10 == 0:
            print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%')
        

    # 测试模型
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        writer.add_scalar('Test Accuracy', accuracy, 0)
        print(f'{model_name} - Test Accuracy: {accuracy * 100:.2f}%')

    writer.close()

# 实例化模型并训练评估
models = {
    'RNN': RNN(),
    'LSTM': LSTM(),
    'GRU': GRU(),
    'BiRNN': BiRNN()
}

for model_name, model in models.items():
    train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test)