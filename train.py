import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import SimpleCNN
from training_monitor import TrainingMonitor

def train_model(epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    monitor = TrainingMonitor()

    model.train()

    for epoch in range(epochs):

        for data, target in loader:

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            monitor.record_loss(loss.item())

        features = monitor.get_features()
        
        return model, features
