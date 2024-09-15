import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import enum

import numpy as np
import PIL.Image as Image

class ModelType(enum.Enum):
    MNIST = "mnist"
    FASHION = "fashion"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.sigmoid(x)
        return out
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        np_target = np.zeros((batch_size, 10), dtype=np.float32)
        for b, t in zip(range(batch_size), target):
                np_target[b, t] = 1.0
        target = torch.from_numpy(np_target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        # loss = F.binary_cross_entropy_with_logits(output, target)
        # loss = F.loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end="")
    print()
            
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target_idx in test_loader:
            batch_size = data.shape[0]
            target_np = np.zeros((batch_size, 10), dtype=np.float32)
            for b, t in zip(range(batch_size), target_idx):
                target_np[b, t] = 1.0
            target = torch.from_numpy(target_np)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction="sum").item()
            # test_loss += F.mse_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            
            for i in range(len(pred)):
                if pred[i] == target_idx[i]:
                    correct += 1
            
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
def main(dataloader, model_type, learning_rate, epochs, batch_size, save_model = False):
    device = "cuda"
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = dataloader("data", train=True, download=True, transform=transform)
    test_dataset = dataloader("data", train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **{"batch_size": batch_size})
    test_loader = torch.utils.data.DataLoader(test_dataset, **{"batch_size": 1000})
    
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    
    for epoch in range(0, epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        print(scheduler.get_last_lr())
        scheduler.step()
    
    if save_model:
        torch.save(model.state_dict(), f"{model_type.value}_fc128_relu_fc10_softmax.pt")


def run_test(model, dataset):
    with torch.no_grad():
        correct = 0
        test_loader = torch.utils.data.DataLoader(dataset, **{"batch_size": len(dataset.targets)})
        
        for idx, (sample, target_ids) in enumerate(test_loader):
            batch_size = sample.shape[0]
            target_np = np.zeros((batch_size, 10), dtype=np.float32)
            for b, t in zip(range(batch_size), target_ids):
                target_np[b, t] = 1.0
            target = torch.from_numpy(target_np)
            data, target = sample.to("cuda"), target.to("cuda")
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            for i in range(len(pred)):
                if pred[i] == target_ids[i]:
                    correct += 1
        
        print(f"Predicted correct: {correct} / {len(dataset.data)} => {correct * 100 / len(dataset.data)} %")


def test_model(model_type, dataloader):
    
    mnist_model_name = "mnist_fc128_relu_fc10_softmax.pt"
    fashion_model_name = "fashion_fc128_relu_fc10_softmax.pt"
    
    model_name = mnist_model_name if model_type == ModelType.MNIST else fashion_model_name
    
    model = Net().to("cuda")
    model.load_state_dict(torch.load(model_name, weights_only=True))    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = dataloader("data", download=True, train=False, transform=transform)
    dataset_sample = Image.fromarray(dataset.data[0].numpy())
    dataset_sample.save(f"{model_type.value}_{dataset.targets[0]}.bmp")
    
    print(f"Validating {model_type.value}:")
    run_test(model, dataset)

if __name__ == "__main__":
    
    # CHANGE THIS LINE TO EITHER USE THE NORMAL OR FASHION MNIST 
    model_type = ModelType.MNIST # one of fashion, mnist
    
    dataset = datasets.MNIST if model_type == ModelType.MNIST else datasets.FashionMNIST
    
    # main(dataset, model_type, learning_rate = 1.0, epochs = 10, batch_size = 32, save_model = False)
    test_model(model_type, dataset)
    # get_loss_val()
    