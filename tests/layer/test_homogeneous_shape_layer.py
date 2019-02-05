from shapenet.layer import HomogeneousShapeLayer
import torch
import numpy as np
import pytest
import warnings


@pytest.mark.parametrize("shapes,n_dims,use_cpp,params,target",
                         [
                             (np.zeros((45, 128, 2)), 2, False,
                              torch.ones(10, 44 + 1 + 1 + 2, 1, 1),
                              torch.ones(10, 128, 2)),
                             (np.zeros((45, 128, 2)), 2, True,
                              torch.ones(10, 44 + 1 + 1 + 2, 1, 1),
                              torch.ones(10, 128, 2)),
                             (np.zeros((45, 128, 3)), 3, False,
                              torch.ones(10, 44 + 3 + 3 + 3, 1, 1),
                              torch.ones(10, 128, 3)),
                             (np.zeros((45, 128, 3)), 3, True,
                              torch.ones(10, 44 + 3 + 3 + 3, 1, 1),
                              torch.ones(10, 128, 3))
                         ]
                         )
def test_homogeneous_shape_layer(shapes, n_dims, use_cpp, params, target):
    layer = HomogeneousShapeLayer(shapes, n_dims, use_cpp)
    params.requires_grad_(True)

    assert (layer(params.float()) == target.float()).all()
    try:
        result = layer(params.float())
        result.backward(torch.ones_like(result))
    except:
        assert False, "Backward not successful"
