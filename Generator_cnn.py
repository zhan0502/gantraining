import torch.nn as nn
import torch.nn.functional as F
class Net1(nn.Module):
    
        def __init__(self):
            super(Net1, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()

            self.pool = nn.MaxPool2d(kernel_size=2)

            self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.relu4 = nn.ReLU()

            self.fc = nn.Linear(in_features=2 * 32 * 32, out_features=3072)
         

        def forward(self, input):
            output = self.conv1(input)
            output = self.relu1(output)

            output = self.conv2(output)
            output = self.relu2(output)

            output = self.pool(output)

            output = self.conv3(output)
            output = self.relu3(output)

            output = self.conv4(output)
            output = self.relu4(output)
            #print(output.shape)
            output = output.view(-1, 2 * 32 * 32)

            output = self.fc(output)
 
            return output