import torch
import torch.nn as nn

# Define a simple network for the 4070 Ti
class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        # Input: [Aggregate Height, Holes, Bumpiness, Completed Lines]
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: The Q-Value (suitability score)
        )

    def forward(self, x):
        return self.fc(x)

# Offload to 4070 Ti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
