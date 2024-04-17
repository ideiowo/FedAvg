import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # Flatten 28x28 to a 784 vector
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  # 10 classes for MNIST/FMNIST

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after first convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after second convolution
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted for flattened size after pooling
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization before the first fully connected layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)  # Adjust view to match the new flattened size
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        # 加載未預訓練的 SqueezeNet 模型
        self.squeezenet = models.squeezenet1_1(pretrained=False)
        # 替換分類器部分以適應目標類別數
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.squeezenet.num_classes = num_classes
    
    def forward(self, x):
        return self.squeezenet(x)

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
if __name__ == "__main__":
    # 創建模型實例
    dnn_model = DNN()
    cnn_model = CNN()
    squeezenet_model = SqueezeNet()

    # 打印各模型的參數數量
    print("DNN Model Parameters:")
    print_model_parameters(dnn_model)

    print("\nCNN Model Parameters:")
    print_model_parameters(cnn_model)

    print("\nSqueezeNet Model Parameters:")
    print_model_parameters(squeezenet_model)
