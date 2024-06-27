import torch
import numpy as np
import pandas as pd
import transformers
import datasets
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score
from datasets import load_dataset

train_processed = torch.load('train_processed.pt')
train_labels = torch.load('train_labels.pt')
test_processed = torch.load('test_processed.pt')
test_labels = torch.load('test_labels.pt')

print(train_processed['pixel_values'].shape)
print(train_labels.shape)
print(test_processed['pixel_values'].shape)
print(test_labels.shape)

train_loader = DataLoader(train_processed, batch_size=16, shuffle=True)
train_labels_loader = DataLoader(train_labels, batch_size=16, shuffle=True)
test_loader = DataLoader(test_processed, batch_size=16)
test_labels_loader = DataLoader(test_labels, batch_size=16)

print("Dataset loaded")

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
    
model = Model().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

losses = []
accuracies = []

print("Training")
correct = 0
total = 0
for epoch in range(1):
    model.train()
    for (batch,label_batch) in (zip(train_processed['pixel_values'], train_labels)):
        optimizer.zero_grad()
        # print(batch.unsqueeze(0).shape)
        # print("Label shape: ", label_batch.shape)
        # print("Label: ", label_batch[0])
        # print("Revised label shape: ", label_batch.unsqueeze(0).shape)
        out = model(batch.unsqueeze(0).to('cuda'))
        predicted_label = torch.argmax(out)
        # print("Predicted label: ", predicted_label)
        # print("Actual label: ", label_batch[0])
        if predicted_label.cpu() == label_batch[0]:
            correct += 1
        total += 1
        accuracies.append(correct/total)
        loss = loss_fn(out, label_batch[0].unsqueeze(0).to('cuda'))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {correct/total}")

torch.save(model, 'cifar-final.pt')
torch.save(losses, 'losses.pt')
torch.save(accuracies, 'accuracies.pt')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (batch, label_batch) in (zip(test_processed['pixel_values'], test_labels)):
        input = batch.unsqueeze(0).to('cuda')
        out = model(input)
        _, predicted = torch.max(out, 1)
        total += label_batch[0].shape[0]
        correct += (predicted == label_batch[0]).sum().item()

print("Accuracy: ", correct/total)