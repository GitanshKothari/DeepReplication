import torch
import torch.nn as nn


class inception_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(inception_conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
    
class inception_block(nn.Module):
    def __init__(self, in_channels, out_1, red_3, out_3, red_5, out_5, out_1_pool):
        super(inception_block, self).__init__()

        self.block1 = inception_conv_block(in_channels, out_1, kernel_size=1)

        self.block2 = nn.Sequential(
            inception_conv_block(in_channels, red_3, kernel_size = 1),
            inception_conv_block(red_3, out_3, kernel_size = 3, stride = 1, padding = 1)
        )

        self.block3 = nn.Sequential(
            inception_conv_block(in_channels, red_5, kernel_size = 1),
            inception_conv_block(red_5, out_5, kernel_size = 5, stride = 1, padding = 2)
        )

        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            inception_conv_block(in_channels, out_1_pool, kernel_size = 1)
        )

    def forward(self, x):
        return torch.cat([self.block1(x), self.block2(x), self.block3(x), self.block4(x)], 1)
    

class GoogLeNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 1000):
        super(GoogLeNet, self).__init__()

        self.conv1 = inception_conv_block(in_channels = in_channels, out_channels=64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding=1)

        self.conv2 = inception_conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

    
    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))

        x = self.maxpool3(self.inception3b(self.inception3a(x)))

        x = self.maxpool4(self.inception4e(self.inception4d(self.inception4c(self.inception4b(self.inception4a(x))))))

        x = self.avgpool(self.inception5b(self.inception5a(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)

        x = self.linear(x)

        return(x)
        

if __name__ == '__main__':
    x = torch.randn(3, 3, 224, 224)
    model = GoogLeNet()
    print(model(x).shape)

