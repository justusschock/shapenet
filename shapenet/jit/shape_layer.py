# author: Justus Schock (justus.schock@rwth-aachen.de)

import numpy as np
import torch


class ShapeLayer(torch.jit.ScriptModule):
    def __init__(self, shapes, use_cpp=False):
        """

        Parameters
        ----------
        shapes : np.ndarray
            the shape components needed by the actual shape layer implementation
        use_cpp : bool
            whether to use cpp implementation or not
            (Currently only the python version is supported)

        """
        super().__init__()

        self._layer = _ShapeLayerPy(shapes)
        assert not use_cpp, "Currently only the Python Version is supported"

    @torch.jit.script_method
    def forward(self, shape_params: torch.Tensor):
        return self._layer(shape_params)

    @property
    def num_params(self):
        return self._layer.num_params


class _ShapeLayerPy(torch.jit.ScriptModule):
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

    @torch.jit.script_method
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
        shapes = shapes.expand(shape_params.size(0), shapes.size(1),
                               shapes.size(2))

        components = getattr(self, "_shape_components")
        components = components.expand(shape_params.size(0),
                                       components.size(1), components.size(2),
                                       components.size(3))

        weighted_components = components.mul(
            shape_params.expand_as(components))

        shapes = shapes.add(weighted_components.sum(dim=1))

        return shapes

    @property
    def num_params(self):
        return getattr(self, "_shape_components").size(1)
