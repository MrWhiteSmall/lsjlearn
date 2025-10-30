
'''
2028MB
bz 128
h=w=32

Training finished. Best test accuracy: 0.8794
'''

import torch
from torch import nn
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(AlexNet,self).__init__()

        # nn.Conv2d()
        # nn.ReLU(inplace=True)
        # nn.BatchNorm2d()
        # nn.MaxPool2d()

        # nn.Linear()

        '''input bz,channel=3,32,32
        1 conv in 3 out 96 k 3 p 1 s 2 => bz 96 16 16
          relu 激活
          norm 

        中间层  信息密度高 的时候，不加池化
        
        2 conv in 96 out 256 k 3 p 1 => bz 256 16 16
          relu
          
        3 conv in 256 out 384 k 3 p 1 => bz 384 16 16
          relu

        4 conv in 384 out 256 k 3 p1 => bz 256 16 16
          relu
          norm
          pool k 3 s 2 => bz 256 8 8 

        classifier
        1 dropout 
          linear in 256*8*8 out 4096
          relu

        2 dropout
          linear in 4096 out 4096
          relu
        
        3 linear in 4096 out num_classes
        '''

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),

            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*8*8, 4096), # torch.Size([128, 12544])
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# INSERT_YOUR_CODE
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

# INSERT_YOUR_CODE
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset_cifar import CIFAR10FromTar

def train_alexnet(
    tar_path='datasets/cifar-10-python.tar.gz',
    num_classes=10,
    batch_size=128,
    num_epochs=20,
    lr=0.01,
    weight_decay=5e-4,
    device=None,
    save_path='alexnet_cifar10.pth'
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据转换
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = CIFAR10FromTar(tar_path, train=True, transform=transform_train)
    test_dataset = CIFAR10FromTar(tar_path, train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # 模型/损失/优化
    model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练循环
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_acc:.4f}")

        scheduler.step()

    print(f"Training finished. Best test accuracy: {best_acc:.4f}")

# 若直接运行此文件则启动训练
if __name__ == "__main__":
    import argparse,os
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=str, default='2', help='CUDA device id')
    args, unknown = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


    train_alexnet()
