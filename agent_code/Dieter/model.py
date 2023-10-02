"""
Deep Q-Learning Model for Bomber-Man
"""

import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """
    DQN MODEL
    """
    gamma = 0.95
    learning_rate = 0.1

    def __init__(self,input_dim,output_dim):
        super(DQN, self).__init__()
        self.model_sequence = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.LeakyReLU(0.1),
                nn.Linear(2048,512),
                 nn.LeakyReLU(0.1),
                nn.Linear(512,128),
                 nn.LeakyReLU(0.1),
                nn.Linear(128,32),
                 nn.LeakyReLU(0.1),
                nn.Linear(32, output_dim),
                )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),self.learning_rate)

    def forward(self,x):
        x = x.to(device)
        return self.model_sequence(x)

class CONV_DQN(nn.Module):
    """
    Convolutional DQN Model
    """
    gamma = 0.99
    learning_rate = 0.003

    def __init__(self,input_dim, output_dim):
        super(CONV_DQN, self).__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels = 6, out_channels = 32, kernel_size = 3, stride = 1,padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size = 2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride = 1, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                #nn.MaxPool2d(kernel_size = 2, stride=2),
                )
        conv_output_size = self.calculate_conv_output_size(input_dim)
        print(conv_output_size)
        self.fc_layers = nn.Sequential(
                nn.Linear(2304,1024),
                #nn.Dropout(0.5),
                nn.PReLU(),
                nn.Linear(1024,output_dim),
                #nn.Dropout(0.2),
                #nn.PReLU(),
                #nn.Linear(384,output_dim),
                )
        self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(),self.learning_rate)

    def calculate_conv_output_size(self, input_dim):
        dummy_input = torch.zeros(1, 6, input_dim, input_dim)
        dummy_output = self.conv_layers(dummy_input)
        dummy_output = dummy_output.view(dummy_output.size(0), -1)  # Flatten the tensor
        conv_output_size = dummy_output.size(1)
        return conv_output_size
        
    def z_score_normalize(self, data):
        mean = torch.mean(data, axis=(0,1,2,3))  # Calculate mean across all channels and positions
        std = torch.std(data, axis=(0,1,2,3))    # Calculate standard deviation across all channels and positions
        normalized_data = (data - mean) / std
        return normalized_data

    def forward(self,x):
        x = x.to(device)
        x = torch.reshape(x,(-1,6,17,17))
        x = self.z_score_normalize(x)
        x = self.conv_layers(x)
        x = x.view(-1,2304)
        x = self.fc_layers(x)
        return x

