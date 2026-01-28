import torch
import torch.nn as nn
import logging


# =========================
# logging configuration
# =========================
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CNN(nn.Module):
    def  __init__(self,num_classed:int=2):
        try :
            super(CNN,self).__init__()
            logger.info("Initializing CNN model...")
            
            # Convolutional layers
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1= nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.relu2=nn.ReLU()
            self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.conv3=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.relu3=nn.ReLU()
            self.pool3=nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Fully connected layers
            # Assuming input images are 224x224, after 3 pooling layers of kernel size 2, the size will be reduced to 28x28
            self.fc1=nn.Linear(64*28*28, 128)
            self.relu_fc1=nn.ReLU()
            self.fc2=nn.Linear(128, num_classed)
        
            logger.info("CNN model initialized successfully.")
            
        except Exception as e:
            logger.error(f"Error initializing CNN model: {e}")
            raise e
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        try:
            logger.info("Performing forward pass through CNN model...")
            # Convolutional layers with activations and pooling
            
            
            # First conv layer
            x=self.conv1(x)
            x=self.relu1(x)
            x=self.pool1(x)
            
            # Second conv layer
            x=self.conv2(x)
            x=self.relu2(x)
            x=self.pool2(x)
            
            # Third conv layer
            x=self.conv3(x)
            x=self.relu3(x)
            x=self.pool3(x)
            
            # Flatten the tensor for fully connected layers
            x=x.view(x.size(0), -1)  # Flatten the tensor
            
            x=self.fc1(x)
            x=self.relu_fc1(x)
            x=self.fc2(x)
            
            logger.info("Forward pass completed successfully.")
            return x
        
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise e