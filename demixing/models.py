import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    """sparse autoencoder: nN -> expanded sparse hidden -> reconstructed nN"""

    def __init__(self, n_neurons, latent_dim):
        super().__init__()
        self.n_neurons = n_neurons
        self.latent_dim = latent_dim

        self.to_latent = nn.Linear(n_neurons, latent_dim)
        self.output = nn.Linear(latent_dim, n_neurons, bias=True)

    def forward(self, x):
        """
        x: (batch, T, n_neurons)
        returns: prediction (batch, T, n_neurons), hidden (batch, T, latent_dim)
        """
        hidden = F.relu(self.to_latent(x))
        pred = F.softplus(self.output(hidden))
        return pred, hidden


class CausalLFADS(nn.Module):
    """causal LFADS: RNN -> sparse bottleneck -> [hidden + stimulus] -> predicted spikes_{t+1}"""

    def __init__(self, n_neurons, latent_dim, rnn_dim=100):
        super().__init__()
        self.n_neurons = n_neurons
        self.latent_dim = latent_dim
        self.rnn_dim = rnn_dim
        self.n_stim_dims = 1

        self.rnn = nn.RNN(n_neurons, rnn_dim, batch_first=True, nonlinearity='tanh')
        self.to_latent = nn.Linear(rnn_dim, latent_dim)
        self.output = nn.Linear(latent_dim + self.n_stim_dims, n_neurons)

    def forward(self, x, stimulus=None, hidden=None):
        """
        x: (batch, T, n_neurons)
        stimulus: (batch, T, n_stim_dims) or None
        returns: prediction, z, hidden
        """
        rnn_out, hidden = self.rnn(x, hidden)
        z = torch.relu(self.to_latent(rnn_out))

        if stimulus is not None:
            decoder_input = torch.cat([z, stimulus], dim=-1)
        else:
            zeros = torch.zeros(*z.shape[:2], self.n_stim_dims, device=z.device)
            decoder_input = torch.cat([z, zeros], dim=-1)

        fr_pred = F.softplus(self.output(decoder_input))
        return fr_pred, z, hidden
