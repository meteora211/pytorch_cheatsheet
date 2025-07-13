import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basic_transformer import *

# Hyperparameters
log_interval = 10
num_classes = 10
max_len = 28
lr = 0.02
num_epochs = 5
batch_size = 50
input_dim = 28  # Each row of the MNIST image
seq_length = max_len  # Number of rows in the MNIST image
model_dim = 128 # hidden size/embedding size
ffn_hidden = 4 * model_dim
eps = 1e-5 

def transformer_collate_fn(batch):
    images, labels = zip(*batch)
    
    sequences = []
    for img in images:
        # img shape: [1, 28, 28] -> [28, 28]
        img = img.squeeze(0)
        sequences.append(img)
    
    sequences = torch.stack(sequences)  # [batch_size, 28, 28]
    labels = torch.tensor(labels)
    # labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    
    return sequences, labels


class MNISTTransformer(nn.Module):
    def __init__(self, attn, hidden_size, ffn_hidden_size):
        super().__init__()
        self.input_projection = nn.Linear(28, hidden_size)

        self.attn = attn
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.input_projection(x)  # [batch_size, 28, hidden_size]

        z = F.layer_norm(x, [self.hidden_size])
        x = self.attn(x)
        x = x + z

        z = F.layer_norm(x, [self.hidden_size])

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        x = x + z

        x = x.mean(dim = 1) # [batch_size, hidden_size]
        x = self.classifier(x) # [batch_size, num_classes]
        return x
        


def test(model, data_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

def train(model, data_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

def main():
    train_kwargs = {'batch_size': batch_size}
    eval_kwargs = {'batch_size': batch_size}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    eval_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, collate_fn=transformer_collate_fn)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, **eval_kwargs, collate_fn=transformer_collate_fn)

    device = torch.device('cuda')

    # Initialize model weights.
    np.random.seed(0)

    # model = MNISTTransformer(BasicAttention(model_dim), model_dim, ffn_hidden).to(device)
    model = MNISTTransformer(MultiHeadAttention(model_dim, 8), model_dim, ffn_hidden).to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr)

    for epoch in range(1, num_epochs+1):
        train(model, train_dataloader, optimizer, device, epoch)
        test(model, eval_dataloader, device)

if __name__ == '__main__':
    main()
