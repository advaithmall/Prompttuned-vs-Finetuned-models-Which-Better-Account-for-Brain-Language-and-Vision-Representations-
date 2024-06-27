from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor
from PIL import Image
import requests

url = 'https://lumiere-a.akamaihd.net/v1/images/darth-vader-main_4560aff7.jpeg?region=71%2C0%2C1139%2C854'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

import torch
from datasets import load_dataset
from tqdm import tqdm
dataset = load_dataset("cifar10")
data = []
for row in tqdm(dataset['train'], total = len(dataset['train'])):
    image = row['img']
    label = row['label']
    image = processor(images=image, return_tensors="pt")
    image = image['pixel_values']
    new_row = [image, label]
    data.append(new_row)

# split into train and test, take 10000 train samples and 1000 test samples
# randomly shuffle the data list
import random
random.shuffle(data)
train_split = data[:10000]
test_split = data[10000:11000]
labels = []
for row in data:
    labels.append(row[1])

print(set(labels))
from torch import nn, optim
class ptuned_VIT(nn.Module):
    def __init__(self, num_classes):
        super(ptuned_VIT, self).__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.embeddings = nn.Embedding(3, 768)
        self.classification_layer = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.trainable = [self.embeddings, self.classification_layer]
        for param in self.model.parameters():
            param.requires_grad = False
    def forward(self, x):
        tens = [0,1,2]
        tens = torch.tensor(tens)
        tens = tens.to(device)
        x1 = self.embeddings(tens)
        x2 = self.model.embeddings(x)
        # change x1 shape from x, 768 to 1, x, 768
        x1 = x1.unsqueeze(0)
        # concat x1 and x2 along the first dimension
        # x1 is 1, 3, 768, make is x2.shape[0], 3, 768
        x1 = x1.expand(x2.shape[0], -1, -1)
        x = torch.cat((x1, x2), 1)
        x = self.model.encoder(x)
        x = x['last_hidden_state']
        x = self.model.pooler(x)
        x = self.classification_layer(x)
        x = self.softmax(x)
        return x
        
classes = len(set(labels))
p_tokens = 3
import torch
from torch import nn, optim
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# use data loader
train_loader = torch.utils.data.DataLoader(train_split, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_split, batch_size=32, shuffle=True)


# use corss entropy loss
# import accuracy and f1 from sklearn
model = ptuned_VIT(classes)
model = model.to(device)
from sklearn.metrics import accuracy_score, f1_score
loss_func = nn.CrossEntropyLoss()
# use adam optimizer
optimizer = optim.Adam(model.parameters(),lr=0.001)
acc_list = []
n_epochs = 10
fin_acc = []
for epoch in range(n_epochs):
    batch_no = 0
    for sample in train_loader:
        image = sample[0]
        label = sample[1]
        image = image.to(device)
        label = label.to(device)
        # iamge shape is a, 1, b, c, d make it a, b, c, d
        image = image.squeeze(1)
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # calculate accuracy
        pred = torch.argmax(output, dim=1)
        acc = accuracy_score(label.cpu(), pred.cpu())
        acc_list.append(acc)
        running_acc = sum(acc_list)/len(acc_list)
        fin_acc.append(acc)
        print(f"Epoch: {epoch}, Batch: {batch_no}/313, Loss: {loss.item()}, Running Accuracy: {running_acc}, Current Accuracy {acc}")
        batch_no += 1
    # write testing loop using test_loader
    acc_list = []
    for sample in test_loader:
        image = sample[0]
        label = sample[1]
        image = image.to(device)
        label = label.to(device)
        image = image.squeeze(1)
        output = model(image)
        pred = torch.argmax(output, dim=1)
        acc = accuracy_score(label.cpu(), pred.cpu())
        acc_list.append(acc)
        running_acc = sum(acc_list)/len(acc_list)
        print(f"Epoch: {epoch}, Test Accuracy: {running_acc}")
    torch.save(model, f"pt_vit_cifar_model_{epoch}.pt")
    torch.save(fin_acc, f"pt_vit_cifar_acc_{epoch}.pt")

