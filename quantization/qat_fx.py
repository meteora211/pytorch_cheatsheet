from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, data_loader, optimizer, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

def test(model, data_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

# hyper params
batch_size = 64
epochs = 1
lr = 1.0
seed = 1

use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
eval_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size)

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr = lr)
scheduler = StepLR(optimizer, step_size = 1)

backend = 'fbgemm'
qconfig = torch.quantization.get_default_qat_qconfig(backend)
qconfig_mapping = {"" : qconfig}
example_inputs = (torch.randn(1,1,28,28),)
model_fx = prepare_qat_fx(model, qconfig_mapping, example_inputs)

for epoch in range(1, epochs+1):
    train(model_fx, train_dataloader, optimizer, device, epoch)
    test(model_fx, eval_dataloader, device)
    scheduler.step()

model_fx = convert_fx(model_fx)
test(model_fx, eval_dataloader, device)
