import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



class Client:
    def __init__(self, client_id, data_loader, model, optimizer, criterion):
        self.client_id = client_id
        self.data_loader = data_loader      
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, epochs=1):
        self.model.to(self.device)  # 确保模型在正确的设备上
        gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}
        model_updates = {name: torch.zeros_like(param).to(self.device) for name, param in self.model.named_parameters()}  # 存储模型更新
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            # 初始化梯度累加器
            for batch_idx, (data, target) in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 累加当前batch的梯度
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad

                # 记录模型更新前的参数
                prev_params = {name: param.clone() for name, param in self.model.named_parameters()}

                self.optimizer.step()

                # 计算并存储模型更新
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        model_updates[name] += prev_params[name] - param

                total_loss += loss.item()

            train_loss = total_loss / len(self.data_loader)


            print(f"Client {self.client_id} - Epoch {epoch+1} - Train Loss: {train_loss}")#, Validation Loss: {val_loss}, Accuracy: {accuracy}%

        return gradients, train_loss#, val_loss, accuracy


    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        return val_loss, accuracy

    def update_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        
        
