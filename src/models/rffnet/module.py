from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class RFFLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sampler: Callable[[int, int], Tensor] = torch.randn,
    ) -> None:
        """Constructor of Random Fourier Features Layer."""
        super(RFFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.relevances = nn.Parameter(torch.empty(in_features))

        # Buffers
        self.register_buffer(
            "_omega_sample",
            sampler(self.in_features, self.out_features),
            persistent=True,
        )
        self.register_buffer(
            "_unif_sample",
            torch.rand(self.out_features) * 2 * np.pi,
            persistent=True,
        )
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass on the Module with input Tensor x.

        Parameters
        ----------
        x : Tensor, shape (n_samples, in_features)

        Returns
        -------
        output : Tensor, shape (n_samples, out_features)
        """
        output = torch.cos((x * self.relevances) @ self._omega_sample + self._unif_sample)
        return output

    def reset_parameters(self, val: float = 0.0) -> None:
        """Reset parameters of the Module. The default is to initialize
        all parameters with a constant value.

        Parameters
        ----------
        val : float (default=0.0)
        """
        nn.init.constant_(self.relevances, val)

    def __repr__(self) -> str:
        """Get string representation of the module.

        Returns
        -------
        repr : str
            String representation.
        """
        return (
            f"RFFLayer(in_features={self.in_features}, out_features={self.out_features})"
        )


class RFFNet(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, int, int],
        sampler: Callable[[int, int], Tensor] = torch.randn,
    ) -> None:
        """Constructor of the Random Fourier Features Network."""
        super(RFFNet, self).__init__()

        self.rff = RFFLayer(dims[0], dims[1], sampler)
        self.linear = nn.Linear(dims[1], dims[2], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass on the Module with input Tensor x.

        Parameters
        ----------
        x : Tensor, shape (n_samples, dims[0])

        Returns
        -------
        output : Tensor, shape (n_samples, dims[-1])
        """
        random_features = self.rff(x)
        return self.linear(random_features)

    def get_relevances(self) -> np.ndarray:
        """Get the relevances parameter from the RFFLayer.

        Returns
        -------
        relevances : np.ndarray, shape (dims[0],)
        """
        return self.rff.relevances.detach().cpu().numpy()
