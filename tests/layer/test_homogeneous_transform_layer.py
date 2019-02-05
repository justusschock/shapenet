from shapenet.layer.homogeneous_transform_layer import \
    HomogeneousTransformationLayer, \
    _HomogeneousTransformationLayerCpp, \
    _HomogeneousTransformationLayerPy
import torch


def test_homogeneous_transform_layer():
    shapes_2d = torch.rand(10, 68, 2, requires_grad=True)
    rotation_params_2d = torch.rand(10, 1, 1, 1, requires_grad=True)
    translation_params_2d = torch.rand(10, 2, 1, 1, requires_grad=True)
    scale_params_2d = torch.rand(10, 1, 1, 1, requires_grad=True)

    layer_2d_py = _HomogeneousTransformationLayerPy(n_dims=2)
    layer_2d_cpp = _HomogeneousTransformationLayerCpp(n_dims=2)

    result_2d_py = layer_2d_py(shapes_2d, rotation_params_2d,
                               translation_params_2d, scale_params_2d)
    result_2d_cpp = layer_2d_cpp(shapes_2d, rotation_params_2d,
                                 translation_params_2d, scale_params_2d)

    assert (result_2d_py - result_2d_cpp).abs().sum() < 1e-6

    try:
        result_2d_py.backward(torch.ones_like(result_2d_py))
    except:
        assert False, "Backward not successful"

    try:
        result_2d_cpp.backward(torch.ones_like(result_2d_cpp))
    except:
        assert False, "Backward not successful"

    layer_2d_py = HomogeneousTransformationLayer(2, False)
    layer_2d_cpp = HomogeneousTransformationLayer(2, True)

    result_2d_py = layer_2d_py(shapes_2d,
                               torch.cat([rotation_params_2d,
                                          translation_params_2d,
                                          scale_params_2d], dim=1))
    result_2d_cpp = layer_2d_cpp(shapes_2d,
                                 torch.cat([rotation_params_2d,
                                            translation_params_2d,
                                            scale_params_2d], dim=1))

    assert (result_2d_py - result_2d_cpp).abs().sum() < 1e-6

    try:
        result_2d_py.backward(torch.ones_like(result_2d_py))
    except:
        assert False, "Backward not successful"

    try:
        result_2d_cpp.backward(torch.ones_like(result_2d_cpp))
    except:
        assert False, "Backward not successful"

    shapes_3d = torch.rand(10, 68, 3, requires_grad=True)
    rotation_params_3d = torch.rand(10, 3, 1, 1, requires_grad=True)
    translation_params_3d = torch.rand(10, 3, 1, 1, requires_grad=True)
    scale_params_3d = torch.rand(10, 3, 1, 1, requires_grad=True)

    layer_3d_py = _HomogeneousTransformationLayerPy(n_dims=3)
    layer_3d_cpp = _HomogeneousTransformationLayerCpp(n_dims=3)

    result_3d_py = layer_3d_py(shapes_3d, rotation_params_3d,
                               translation_params_3d, scale_params_3d)
    result_3d_cpp = layer_3d_cpp(shapes_3d, rotation_params_3d,
                                 translation_params_3d, scale_params_3d)

    assert (result_3d_py - result_3d_cpp).abs().sum() < 1e-6

    try:
        result_3d_py.backward(torch.ones_like(result_3d_py))
    except:
        assert False, "Backward not successful"

    try:
        result_3d_cpp.backward(torch.ones_like(result_3d_cpp))
    except:
        assert False, "Backward not successful"

    layer_3d_py = HomogeneousTransformationLayer(3, False)
    layer_3d_cpp = HomogeneousTransformationLayer(3, True)

    result_3d_py = layer_3d_py(shapes_3d,
                               torch.cat([rotation_params_3d,
                                          translation_params_3d,
                                          scale_params_3d], dim=1))
    result_3d_cpp = layer_3d_cpp(shapes_3d,
                                 torch.cat([rotation_params_3d,
                                            translation_params_3d,
                                            scale_params_3d], dim=1))

    assert (result_3d_py - result_3d_cpp).abs().sum() < 1e-6

    try:
        result_3d_py.backward(torch.ones_like(result_3d_py))
    except:
        assert False, "Backward not successful"

    try:
        result_3d_cpp.backward(torch.ones_like(result_3d_cpp))
    except:
        assert False, "Backward not successful"
