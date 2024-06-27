import torch
from transformers import ViTImageProcessor, ViTModel
import torch.nn as nn
import numpy as np
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.model(x).last_hidden_state[:,0]
        x = self.fc(x)
        x = self.fc2(x)
        return x

model = torch.load('mnist-final.pt')
print(model)

accuracies = torch.load('accuracies-mnist.pt')
accuracies = np.array(accuracies)
accuracy  = np.mean(accuracies)
print("Accuracy: " , accuracy)


