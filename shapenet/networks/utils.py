# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch


class CustomGroupNorm(torch.nn.Module):
    """
    Custom Group Norm which adds n_groups=2 as default parameter
    """

    def __init__(self, n_features, n_groups=2):
        """

        Parameters
        ----------
        n_features : int
            number of input features
        n_groups : int
            number of normalization groups
        """
        super().__init__()
        self.norm = torch.nn.GroupNorm(n_groups, n_features)

    def forward(self, x):
        """
        Forward batch through network

        Parameters
        ----------
        x : torch.Tensor
            batch to forward

        Returns
        -------
        torch.Tensor
            normalized results

        """
        return self.norm(x)
