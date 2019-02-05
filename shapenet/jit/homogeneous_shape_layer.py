# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
from .shape_layer import ShapeLayer
from .homogeneous_transform_layer import HomogeneousTransformationLayer


class HomogeneousShapeLayer(torch.jit.ScriptModule):
    """
    Module to Perform a Shape Prediction
    (including a global homogeneous transformation)

    """

    def __init__(self, shapes, n_dims, use_cpp=False):
        """

        Parameters
        ----------
        shapes : np.ndarray
            shapes to construct a :class:`ShapeLayer`
        n_dims : int
            number of shape dimensions
        use_cpp : bool
            whether or not to use (experimental) C++ Implementation

        See Also
        --------
        :class:`ShapeLayer`
        :class:`HomogeneousTransformationLayer`

        """
        super().__init__()

        self._shape_layer = ShapeLayer(shapes, use_cpp)
        self._homogen_trafo = HomogeneousTransformationLayer(n_dims, use_cpp)

        self.register_buffer("_indices_shape_params",
                             torch.arange(self._shape_layer.num_params))
        self.register_buffer("_indices_homogen_params",
                             torch.arange(self._shape_layer.num_params,
                                          self.num_params))

    @torch.jit.script_method
    def forward(self, params: torch.Tensor):
        """
        Performs the actual prediction

        Parameters
        ----------
        params : :class:`torch.Tensor`
            input parameters

        Returns
        -------
        :class:`torch.Tensor`
            predicted shape

        """

        shape_params = params.index_select(
            dim=1, index=getattr(self, "_indices_shape_params")
        )

        transformation_params = params.index_select(
            dim=1, index=getattr(self, "_indices_homogen_params")
        )
        shapes = self._shape_layer(shape_params)
        transformed_shapes = self._homogen_trafo(shapes, transformation_params)

        return transformed_shapes

    @property
    def num_params(self):
        """
        Property to access these layer's number of parameters

        Returns
        -------
        int
            number of parameters

        """
        return self._shape_layer.num_params + self._homogen_trafo.num_params
