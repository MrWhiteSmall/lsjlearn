'''
Epoch [40/40]  Loss: 0.2033  Acc: 0.9295
Test Accuracy: 0.8809
Training finished. Best test accuracy: 0.8817
'''



import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,stride=1) -> None:
        super(ResBlock,self).__init__()

        '''
        conv1 in_c out_c k 3 s stride p 1 
        norm

        conv out_c out_c k 3 s 1 p 1
        norm
        '''

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out

# AlexNet 池化 k 11 -> k 5 -> k 3
# ResNet stride 2 2 2 ...
'''input bz 3 32 32
ResNet
features
    resblock in 3 out 96 s 2 => bz 96 16 16
    resblock in 96 out 256 s 2 => bz 256 8 8 
    resblock in 256 out 384 s 1 => bz 384 8 8  
    resblock in 384 out 256 s 1 => bz 256 8 8

classifier
    dropout
    linear in 256*8*8 out 4096
    relu

    dropout
    linear in 4096 out 4096
    relu

    linear in 4096 out num_classes
'''
class ResNet(nn.Module):
    def __init__(self, num_classes=10) -> None:
        super(ResNet,self).__init__()

        # INSERT_YOUR_CODE
        # ResNet for CIFAR-10, following provided structure
        self.layer1 = ResBlock(3, 96, stride=2)
        self.layer2 = ResBlock(96, 256, stride=2)
        self.layer3 = ResBlock(256, 384, stride=1)
        self.layer4 = ResBlock(384, 256, stride=1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*8*8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)  # 默认10类
        )

    def forward(self, x):
        # x: [bz, 3, 32, 32]
        x = self.layer1(x) # [bz, 96, 16, 16]
        x = self.layer2(x) # [bz, 256, 8, 8]
        x = self.layer3(x) # [bz, 384, 8, 8]
        x = self.layer4(x) # [bz, 256, 8, 8]
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os

# INSERT_YOUR_CODE
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset_cifar import CIFAR10FromTar

def train_resnet(
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

    # INSERT_YOUR_CODE

    # 模型/损失/优化
    model = ResNet(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

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
            torch.save(model.state_dict(), 'resnet_cifar10.pth')
            print(f"Best model saved with accuracy: {best_acc:.4f}")

        scheduler.step()

    print(f"Training finished. Best test accuracy: {best_acc:.4f}")

