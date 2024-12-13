import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torchvision.models import vgg16, vgg19
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 資料預處理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG 模型需要 224x224 輸入
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加載 CIFAR-100 資料集
    train_dataset = CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4,persistent_workers=True)  # Increase num_workers as needed
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,persistent_workers=True)  # Increase num_workers as needed

    # 定義 PyTorch Lightning 模型
    class VGGClassifier(pl.LightningModule):
        def __init__(self, model_type='vgg16'):
            super(VGGClassifier, self).__init__()
            if model_type == 'vgg16':
                self.model = vgg16(pretrained=True)
            elif model_type == 'vgg19':
                self.model = vgg19(pretrained=True)

            # 替換分類層
            self.model.classifier[6] = torch.nn.Linear(4096, 100)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-4)

    # 初始化模型
    model = VGGClassifier(model_type='vgg16')  # 或者 'vgg19'

    # 訓練模型
    trainer = pl.Trainer(max_epochs=10, accelerator='cpu')
    trainer.fit(model, train_loader, test_loader)

    # 切換到評估模式
    model.eval()

    # 獲取測試數據的一批
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 對一批數據進行推論
    with torch.no_grad():
        outputs = model(images)  # 如果使用 GPU，則使用 images.cuda()
        _, preds = torch.max(outputs, 1)

    # 定義 CIFAR-100 的類別名稱
    classes = train_dataset.classes

    # 可視化前5張測試圖片及預測結果
    for i in range(5):
        # 從張量轉換為可視化的圖片
        img = images[i].permute(1, 2, 0).cpu().numpy()  # 調整通道順序
        img = np.clip((img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)  # 反正規化

        # 顯示圖片
        plt.imshow(img)
        plt.title(f"實際: {classes[labels[i].item()]}\n預測: {classes[preds[i].item()]}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()
