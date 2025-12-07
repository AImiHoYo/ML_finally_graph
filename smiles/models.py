import torch
from torch.nn import Linear, Module, Embedding, GRU, Sequential, ReLU, Dropout

class SmilesRNN(Module):
    """An RNN block for processing a SMILES sequence."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SmilesRNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)

class SmilesPredictor(Module):
    """The complete RNN model for SMILES sequences."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SmilesPredictor, self).__init__()
        self.drug_rnn = SmilesRNN(vocab_size, embedding_dim, hidden_dim)
        self.solvent_rnn = SmilesRNN(vocab_size, embedding_dim, hidden_dim)
        
        self.regressor_head = Sequential(
            Linear(hidden_dim * 2 + 1, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 64),
            ReLU(),
            Dropout(0.5)
        )
        self.output_layer = Linear(64, 1)

    def forward(self, drug_seq, solvent_seq, temp, extract_features=False):
        drug_embedding = self.drug_rnn(drug_seq)
        solvent_embedding = self.solvent_rnn(solvent_seq)
        
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)
        x = torch.cat([drug_embedding, solvent_embedding, temp], dim=1)

        features = self.regressor_head(x)
        if extract_features:
            return features
        return self.output_layer(features)
