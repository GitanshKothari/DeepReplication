import torch
import torch.nn as nn
from torchsummary import summary

"""
    This code defines the LeNet architecture for image classification. 
    It consists of two convolutional layers, followed by three fully connected layers. 
    The ReLU activation function is used throughout the network. 
    The forward method defines the flow of data through the network. 
    The input image is first passed through the convolutional layers, then the output is flattened and passed through the fully connected layers. 
    Finally, the output layer produces the classification scores.
"""
class LeNet(nn.Module):
    def __init__(self):
        """
        Initializes the LeNet class.
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5, stride = 1, padding=0)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.avg_pool(self.relu(self.conv1(x)))
        out = self.avg_pool(self.relu(self.conv2(out)))
        out = out.reshape(out.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
if __name__ == '__main__':
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    summary(model, input_size=(1, 32, 32))

