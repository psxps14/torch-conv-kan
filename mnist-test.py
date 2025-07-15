from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn

from kan_convs import KANConv2DLayer

import torch.optim as optim


class SimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            groups: int = 1):
        super(SimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], spline_order=spline_order, kernel_size=3, groups=1, padding=1, stride=1, dilation=1),
            KANConv2DLayer(layer_sizes[0], layer_sizes[1], spline_order=spline_order, kernel_size=3, groups=groups, padding=1, stride=2, dilation=1),
            KANConv2DLayer(layer_sizes[1], layer_sizes[2], spline_order=spline_order, kernel_size=3, groups=groups, padding=1, stride=2, dilation=1),
            KANConv2DLayer(layer_sizes[2], layer_sizes[3], spline_order=spline_order, kernel_size=3, groups=groups, padding=1, stride=1, dilation=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[3], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

layer_sizes = [32, 64, 128, 256]
model = SimpleConvKAN(layer_sizes=layer_sizes, num_classes=10, input_channels=1, spline_order=3, groups=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3 #5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    batch = 1
    for images, labels in train_loader:
        print(f"batch: {batch}", flush=True)

        batch += 1
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

print("made it to the end of the file")