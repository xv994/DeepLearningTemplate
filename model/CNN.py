import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = self.conv1(x)   # [batch_size, 32, 28, 28]
        x = self.relu(x)    # [batch_size, 32, 28, 28]
        x = self.maxpool(x) # [batch_size, 32, 14, 14]
        
        x = self.conv2(x)   # [batch_size, 64, 14, 14]
        x = self.relu(x)    # [batch_size, 64, 14, 14]
        x = self.maxpool(x) # [batch_size, 64, 7, 7]
        
        x = x.view(x.size(0), -1) # [batch_size, 64*7*7]
        x = self.fc1(x)           # [batch_size, 128]
        x = self.relu(x)          # [batch_size, 128]
        
        x = self.fc2(x)           # [batch_size, 10]
        x = self.softmax(x)       # [batch_size, 10]
        
        return x