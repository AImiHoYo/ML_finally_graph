from torch.nn import Linear, Module, Sequential, ReLU, Dropout

class SolubilityPredictor(Module):
    """A standard Feed-Forward Neural Network for tabular data."""
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256):
        super(SolubilityPredictor, self).__init__()
        self.layer1 = Linear(input_size, hidden_size1)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(0.5)
        self.feature_extractor = Sequential(
            self.layer1, self.relu1, self.dropout1
        )
        self.layer2 = Linear(hidden_size1, hidden_size2)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(0.5)
        self.regressor_head = Sequential(
            self.layer2, self.relu2, self.dropout2
        )
        self.output_layer = Linear(hidden_size2, 1)

    def forward(self, x, extract_features=False):
        features = self.regressor_head(self.feature_extractor(x))
        if extract_features:
            return features
        return self.output_layer(features)
