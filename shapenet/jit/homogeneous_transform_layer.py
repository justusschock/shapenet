# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
import os
from torch.utils.cpp_extension import load as load_cpp


class HomogeneousTransformationLayer(torch.jit.ScriptModule):
    """
    Wrapper Class to Wrap the Python and C++ API into a combined python API

    """
    def __init__(self, n_dims: int, use_cpp=False):
        """

        Parameters
        ----------
        n_dims : int
            number of dimensions
        use_cpp : bool
            whether or not to use C++ implementation

        Raises
        ------
        AssertionError
            if ``use_cpp`` is True, currently only the python version is
            supported

        """

        assert use_cpp==False, "Currently only the python version is supported"
        super().__init__()

        self._n_params = {}

        if n_dims == 2:
            self._n_params["scale"] = 1
            self._n_params["rotation"] = 1
            self._n_params["translation"] = 2
        elif n_dims == 3:
            self._n_params["scale"] = 3
            self._n_params["rotation"] = 3
            self._n_params["translation"] = 3

        self._layer = _HomogeneousTransformationLayerPy(n_dims)

        total_params = 0
        for key, val in self._n_params.items():
            self.register_buffer("_indices_%s_params" % key,
                                 torch.arange(total_params, total_params + val)
                                 )
            total_params += val

    @torch.jit.script_method
    def forward(self, shapes: torch.Tensor, params: torch.Tensor):
        """
        Selects individual parameters from ``params`` and forwards them through 
        the actual layer implementation
        
        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes to transform
        params : :class:`torch.Tensor`
            parameters specifying the affine transformation
        
        Returns
        -------
        Returns
        -------
        :class:`torch.Tensor`
            the transformed shapes in cartesian coordinates

        """

        rotation_params = params.index_select(
            dim=1, index=getattr(self, "_indices_rotation_params")
        )
        scale_params = params.index_select(
            dim=1, index=getattr(self, "_indices_scale_params")
        )
        translation_params = params.index_select(
            dim=1, index=getattr(self, "_indices_translation_params")
        )

        return self._layer(shapes, rotation_params, translation_params,
                           scale_params)

    @property
    def num_params(self):
        num_params = 0
        for key, val in self._n_params.items():
            num_params += val

        return num_params


class _HomogeneousTransformationLayerPy(torch.jit.ScriptModule):
    """
    Module to perform homogeneous transformations in 2D and 3D
    (Implemented in Python)

    """

    __constants__ = ["_n_dims"]

    def __init__(self, n_dims):
        """

        Parameters
        ----------
        n_dims: int
            number of dimensions
        """
        super().__init__()

        homogen_trafo = torch.zeros(1, n_dims+1, n_dims+1)
        homogen_trafo[:, -1, :-1] = 0.
        homogen_trafo[:, -1, -1] = 1.

        self.register_buffer("_trafo_matrix", homogen_trafo)
        self._n_dims = n_dims

    @torch.jit.script_method
    def forward(self, shapes: torch.Tensor, rotation_params: torch.Tensor,
                translation_params: torch.Tensor, scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix and applies it to the
        shape tensor

        Parameters
        ----------
        shapes : :class:`torch.Tensor`
            shapes to transform
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            the transformed shapes in cartesian coordinates

        """

        assert shapes.size(-1) == self._n_dims, "Layer for other " \
                                                "dimensionality specified"

        trafo_matrix = self._ensemble_trafo(rotation_params,
                                            translation_params, scale_params)

        homogen_shapes = torch.cat([shapes,
                                    torch.ones([shapes.size(0),
                                               shapes.size(1), 1],
                                               dtype=shapes.dtype,
                                               device=shapes.device)],
                                   dim=-1)

        transformed_shapes = torch.bmm(homogen_shapes,
                                       trafo_matrix.permute(0, 2, 1))

        return transformed_shapes[:, :, :-1]

    @torch.jit.script_method
    def _ensemble_trafo(self, rotation_params: torch.Tensor,
                        translation_params: torch.Tensor,
                        scale_params: torch.Tensor):
        """
        ensembles the transformation matrix in 2D and 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one per DoF)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (one per dimension)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            transformation matrix

        """

        rotation_params = rotation_params.view(rotation_params.size()[:2])
        translation_params = translation_params.view(
            translation_params.size()[:2])
        scale_params = scale_params.view(scale_params.size()[:2])

        if self._n_dims == 2:
            trafo = self._ensemble_2d_matrix(rotation_params,
                                            translation_params, scale_params)
        else:
            trafo = self._ensemble_3d_matrix(rotation_params,
                                            translation_params, scale_params)

        return trafo

    @torch.jit.script_method
    def _ensemble_2d_matrix(self, rotation_params: torch.Tensor,
                            translation_params: torch.Tensor,
                            scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 2D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (one parameter)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (two parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            2D transformation matrix

        """

        homogen_trafo = getattr(self, "_trafo_matrix").repeat(
            scale_params.size(0), 1, 1).clone()

        homogen_trafo[:, 0, 0] = (scale_params *
                                  rotation_params.cos())[:, 0].clone()
        # s*sin\theta
        homogen_trafo[:, 0, 1] = (scale_params *
                                  rotation_params.sin())[:, 0].clone()
        # -s*sin\theta
        homogen_trafo[:, 1, 0] = (-scale_params *
                                  rotation_params.sin())[:, 0].clone()
        # s*cos\theta
        homogen_trafo[:, 1, 1] = (scale_params *
                                  rotation_params.cos())[:, 0].clone()

        # translation params
        homogen_trafo[:, :-1, -1] = translation_params.clone()

        return homogen_trafo

    @torch.jit.script_method
    def _ensemble_3d_matrix(self, rotation_params: torch.Tensor,
                            translation_params: torch.Tensor,
                            scale_params: torch.Tensor):
        """
        ensembles the homogeneous transformation matrix for 3D

        Parameters
        ----------
        rotation_params : :class:`torch.Tensor`
            parameters specifying the rotation (three parameters)
        translation_params : :class:`torch.Tensor`
            parameters specifying the translation (three parameters)
        scale_params : :class:`torch.Tensor`
            parameter specifying the global scaling factor (one parameter)
            (currently only isotropic scaling supported)

        Returns
        -------
        :class:`torch.Tensor`
            3D transformation matrix

        """

        homogen_trafo = getattr(self, "_trafo_matrix").repeat(
            scale_params.size(0), 1, 1).clone()

        roll = rotation_params[:, 2].unsqueeze(-1)
        pitch = rotation_params[:, 1].unsqueeze(-1)
        yaw = rotation_params[:, 0].unsqueeze(-1)

        # Note that the elements inside the transformation matrix are swapped
        # due to the zyx convention

        # s*(cos(pitch)*cos(roll))
        homogen_trafo[:, 0, 0] = (scale_params *
                                  (pitch.cos() * roll.cos()))[:, 0].clone()

        # s*(cos(pitch)*sin(roll))
        homogen_trafo[:, 0, 1] = (scale_params *
                                  (pitch.cos() * roll.sin()))[:, 0].clone()

        # s*(-sin(pitch))
        homogen_trafo[:, 0, 2] = (scale_params *
                                  (-pitch.sin()))[:, 0].clone()

        # s*(sin(yaw)*sin(pitch)*cos(roll) - cos(yaw)*sin(roll))
        homogen_trafo[:, 1, 0] = (scale_params *
                                  (yaw.sin() * pitch.sin() * roll.cos() -
                                   yaw.cos() * roll.sin()))[:, 0].clone()

        # s*(sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll))
        homogen_trafo[:, 1, 1] = (scale_params *
                                  (yaw.sin() * pitch.sin() * roll.sin() +
                                   yaw.cos() * roll.cos()))[:, 0].clone()

        # s*(sin(yaw)*cos(pitch))
        homogen_trafo[:, 1, 2] = (scale_params *
                                  (yaw.sin() * pitch.cos()))[:, 0].clone()

        # s*(cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll))
        homogen_trafo[:, 2, 0] = (scale_params *
                                  (yaw.cos() * pitch.sin() * roll.cos() +
                                   yaw.sin() * roll.sin()))[:, 0].clone()

        # s*(cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll))
        homogen_trafo[:, 2, 1] = (scale_params *
                                  (yaw.cos() * pitch.sin() * roll.sin() -
                                   yaw.sin() * roll.cos()))[:, 0].clone()

        # s*(cos(yaw)*cos(pitch))
        homogen_trafo[:, 2, 2] = (scale_params *
                                  (yaw.cos() * pitch.cos()))[:, 0].clone()

        # translation params
        homogen_trafo[:, :-1, -1] = translation_params.clone()

        return homogen_trafo


if __name__ == '__main__':
    shapes_2d = torch.rand(10, 68, 2)
    rotation_params_2d = torch.rand(10, 1, 1, 1)
    # translation_params_2d = torch.rand(10, 2, 1, 1)
    translation_params_2d = torch.rand(10, 2, 1, 1)
    scale_params_2d = torch.rand(10, 1, 1, 1)

    print("Creating Python Layer")
    layer_2d_py = _HomogeneousTransformationLayerPy(n_dims=2)


    result_2d_py = layer_2d_py(shapes_2d, rotation_params_2d, translation_params_2d,
                               scale_params_2d)
    shapes_3d = torch.rand(10, 68, 3)
    rotation_params_3d = torch.rand(10, 3, 1, 1)
    # rotation_params_3d = torch.zeros(10, 3, 1, 1)
    translation_params_3d = torch.rand(10, 3, 1, 1)
    # translation_params_3d = torch.zeros(10, 3, 1, 1)
    scale_params_3d = torch.rand(10, 3, 1, 1)

    layer_3d_py = _HomogeneousTransformationLayerPy(n_dims=3)
    result_3d_py = layer_3d_py(shapes_3d, rotation_params_3d, translation_params_3d,
                               scale_params_3d)

