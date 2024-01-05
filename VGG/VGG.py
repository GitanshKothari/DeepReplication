import torch
import torch.nn as nn
from torchsummary import summary


VGG16_acrhitecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, x):
        x = self.create_conv_layers(VGG16_acrhitecture)(x)
        x = x.view(-1, 7*7*512)
        x = self.create_fcs()(x)

        return x
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers.append(nn.Conv2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size=3,
                                        stride = 1,
                                        padding=1))
                layers.append(nn.ReLU())
                in_channels = out_channels
            
            elif type(x) == str:
                if x == "M":
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def create_fcs(self):
        fc_inputs = 7 * 7 * 512 # size of the output from previous layer
        hidden_units = 4096
        dropout_rate = 0.5
        layers = []
        layers.append(nn.Linear(fc_inputs, hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_units, hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_units, self.num_classes))
        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = VGG_net(3, 10).to('cuda')
    input_size = (3, 224, 224)
    dummy_input = torch.randn(*input_size).to('cuda')
    print(f"Tensor type: {dummy_input.type()}")
    if next(model.parameters()).is_cuda:
        print("Model is on CUDA.")
    else:
        print("Model is on CPU.")
    summary(model, input_size=input_size)