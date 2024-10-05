import torch
from torch import nn
from torch.nn import functional

class ConvNet(nn.Module):
    def __init__(
            self,
            X,
            y,
            #hidden_layer_sizes,
            init_scale=1,
            drop_out=0.05,
            kernel_size=3,
            device=None
        ):
        super().__init__()

        #self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device
        self.init_scale = init_scale
        self.kernel_size = kernel_size
        self.num_classes = self._output_shape(y)
        self._build(X.shape[2], X.shape[3], self.num_classes)

    def _output_shape(self, y):
        return torch.unique(y).size(0)

    def _init(self, weight):
        nn.init.normal_(weight, mean=0, std=self.init_scale)

    def _nonlinearity(self):
        return nn.ReLU(inplace=True)
    
    def _build(self, height, width, out_dim):
        #in_dim = height * width
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=self.kernel_size,
                padding=1
            )
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=self.kernel_size,
                padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        #in_dim_side = in_dim**(1/2)
        linear_in_dim = 8 * int(height / 4) * int(width / 4)
        layer_sizes = [linear_in_dim, linear_in_dim >> 1, linear_in_dim >> 2] + [out_dim]
        layers = [self.conv1,
                  self.bn1,
                  self.pool,
                  nn.ReLU(inplace=True),
                  self.conv2,
                  self.bn2,
                  self.pool,
                  nn.ReLU(inplace=True),
                  nn.Flatten()]
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            lin = nn.Linear(in_size, out_size, device=self.device)
            self._init(lin.weight)
            layers.append(lin)
            layers.append(self._nonlinearity())
        layers.pop(-1)
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers.forward(X)
