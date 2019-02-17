import torch
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(4, 32,
            kernel_size=8, stride=4, padding=4, bias=True)
        self.bn1 = nn.BatchNorm2d(32)

        self.maxpool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64,
            kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(5 * 5 * 64, 512)

        self.fc2 = nn.Linear(512, 2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x =self.maxpool(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)

        return x

if __name__ == "__main__":
    model = MyModel()
    print (model)
