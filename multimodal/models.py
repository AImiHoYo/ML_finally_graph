import torch.nn as nn

class MultimodalPredictor(nn.Module):
    """A simple Feed-Forward Neural Network for multimodal features."""
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64):
        super(MultimodalPredictor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size2, 1)
        )

    def forward(self, x):
        return self.regressor(x)
