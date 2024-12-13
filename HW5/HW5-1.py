import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 1. 載入資料集
df = pd.read_csv('./HW5/Iris.csv')
df = df.drop(columns=['Id'])
log_dir = "./HW5/runs/test"
# 2. 資料前處理：LabelEncoder
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

# 3. 分離特徵與目標
iris_x = df.drop(columns=['Species']).values
iris_y = df['Species'].values

# 4. 正規化
scaler = StandardScaler()
iris_x = scaler.fit_transform(iris_x)

# 5. 資料分割
X_train, X_val, y_train, y_val = train_test_split(iris_x, iris_y, test_size=0.2, random_state=42)

# 6. 轉換成tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# 7. Dataset和DataLoader
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = IrisDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = IrisDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 8. 建立模型
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 輸入層
        self.fc2 = nn.Linear(16, 3)  # 輸出層，3類別
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisModel()

# 9. 定義損失函數、優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 10. 訓練模型
writer = SummaryWriter(log_dir=log_dir)  # TensorBoard紀錄
epochs = 600
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)

    # 驗證集準確率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')

writer.close()

# 11. 使用TensorBoard
# 在命令行中啟動 TensorBoard：tensorboard --logdir=runs
