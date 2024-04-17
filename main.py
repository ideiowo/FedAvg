import torch
import torch.nn as nn
import torch.optim as optim
from client import Client
from server import Server
from utils.data_utils import load_MNIST
from models.architecture import DNN,CNN

# 聯邦學習設定參數
NUM_CLIENTS = 10
ROUNDS = 100
EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 初始化裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加載數據
client_loaders, client_test_loaders = load_MNIST(num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE)

# 初始化全局模型
initial_model = DNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(initial_model.parameters(), lr=LEARNING_RATE)

# 創建服務器實例
server = Server(initial_model)

# 創建客戶端實例
clients = []
for i in range(NUM_CLIENTS):
    client_model = DNN().to(device)
    client_model.load_state_dict(initial_model.state_dict())
    client_optimizer = optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
    client_optimizer.load_state_dict(optimizer.state_dict())
    client = Client(client_id=i, data_loader=client_loaders[i], model=client_model, optimizer=client_optimizer, criterion=criterion)
    clients.append(client)

# 訓練迴圈
for round in range(ROUNDS):
    print(f"Round {round + 1}/{ROUNDS}")
    client_gradients = []

    # 訓練每個客戶端並收集梯度
    for client_id, client in enumerate(clients):
        print(f"Training client {client_id + 1}...")
        gradients, train_loss = client.train(epochs=EPOCHS)
        client_gradients.append(gradients)

    # 聚合梯度更新全局模型
    server.aggregate(client_gradients, lr=LEARNING_RATE)

    # 將更新後的全局模型分發給每個客戶端
    global_model = server.get_global_model()


    for client in clients:
        client.update_model(global_model)

    # 驗證全局模型性能
    global_model.eval()  # 設置模型為評估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in client_test_loaders:  # 使用驗證加載器來測試
            data, labels = data.to(device), labels.to(device)
            outputs = global_model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / total
    print(f"驗證結果：準確率 {accuracy}%, 平均損失 {average_loss}")
