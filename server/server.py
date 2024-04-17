import torch        
import copy
import pickle
import numpy as np

class Server:
    def __init__(self, initial_model):
        self.global_model = initial_model

    
    def aggregate(self, client_gradients, lr=0.001):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_clients = len(client_gradients)
        if client_gradients:
            aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items()}
        else:
            print("警告：client_gradients 列表是空的！")
            return
        # 初始化聚合梯度
        
        aggregated_gradients = {name: torch.zeros_like(gradient).to(device) for name, gradient in client_gradients[0].items()}

        # 对选中的客户端梯度进行聚合
        for gradients in client_gradients:
            for name, gradient in gradients.items():
                aggregated_gradients[name] += gradient / num_clients

        # 更新全局模型的权重
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_gradients:
                    param -= lr * aggregated_gradients[name]


    def get_global_model(self):
        """
        获取经过聚合的全局模型。
        """
        if self.global_model is None:
            raise ValueError("Global model is not set or aggregated yet.")
        return self.global_model
