# author: Justus Schock (justus.schock@rwth-aachen.de)

import numpy as np
import torch
import os
from torch.utils.cpp_extension import load as load_cpp


class ShapeLayer(torch.nn.Module):
    """
    Wrapper to compine Python and C++ Implementation under Single API

    """
    def __init__(self, shapes, use_cpp=False):
        """

        Parameters
        ----------
        shapes : np.ndarray
            the actual shape components
        use_cpp : bool
            whether or not to use the (experimental) C++ Implementation
        """
        super().__init__()

        if use_cpp:
            self._layer = _ShapeLayerCpp(shapes)
        else:
            self._layer = _ShapeLayerPy(shapes)

    def forward(self, shape_params: torch.Tensor):
        """
        Forwards parameters to Python or C++ Implementation

        Parameters
        ----------
        shape_params : torch.Tensor
            parameters for shape ensembling

        Returns
        -------
        torch.Tensor
            Ensempled Shape

        """
        return self._layer(shape_params)

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return self._layer.num_params


class _ShapeLayerPy(torch.nn.Module):
    """
    Python Implementation of Shape Layer

    """
    def __init__(self, shapes):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()

        self.register_buffer("_shape_mean", torch.from_numpy(
            shapes[0]).float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(
                _shape).float().unsqueeze(0))

        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer("_shape_components", component_tensor)

    def forward(self, shape_params: torch.Tensor):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : torch.Tensor
            shape parameters

        Returns
        -------
        torch.Tensor
            ensembled shape

        """

        shapes = getattr(self, "_shape_mean").clone()
        shapes = shapes.expand(shape_params.size(0), *shapes.size()[1:])

        components = getattr(self, "_shape_components")
        components = components.expand(shape_params.size(0),
                                       *components.size()[1:])

        weighted_components = components.mul(
            shape_params.expand_as(components))

        shapes = shapes.add(weighted_components.sum(dim=1))

        return shapes

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return getattr(self, "_shape_components").size(1)


class _ShapeLayerCpp(torch.nn.Module):
    """
    C++ Implementation of Shape Layer

    """
    def __init__(self, shapes, verbose=True):
        """

        Parameters
        ----------
        shapes : np.ndarray
            eigen shapes (obtained by PCA)

        """
        super().__init__()

        self.register_buffer("_shape_mean",
                             torch.from_numpy(shapes[0]).float().unsqueeze(0))
        components = []
        for i, _shape in enumerate(shapes[1:]):
            components.append(torch.from_numpy(_shape).float().unsqueeze(0))

        component_tensor = torch.cat(components).unsqueeze(0)
        self.register_buffer("_shape_components", component_tensor)
        self._func = load_cpp("shape_function",
                              sources=[os.path.join(os.path.split(__file__)[0],
                                                    "shape_layer.cpp")],
                              verbose=verbose)

    def forward(self, shape_params: torch.Tensor):
        """
        Ensemble shape from parameters

        Parameters
        ----------
        shape_params : torch.Tensor
            shape parameters

        Returns
        -------
        torch.Tensor
            ensembled shape
        """

        shapes = self._func.forward(shape_params,
                                    getattr(self, "_shape_mean"),
                                    getattr(self, "_shape_components"))

        return shapes

    @property
    def num_params(self):
        """
        Property to access these layer's parameters

        Returns
        -------
        int
            number of parameters

        """
        return getattr(self, "_shape_components").size(1)

