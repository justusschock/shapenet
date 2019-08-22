from shapenet.networks import SingleShapeNetwork
from shapenet.layer import HomogeneousShapeLayer
import torch
import numpy as np
import pytest
from delira.utils.context_managers import DefaultOptimWrapperTorch


@pytest.mark.parametrize("feature_extractor,num_params,in_channels,norm_type,\
                        img_size ",
                         [
                             ("custom", 20, 1, "instance", 224),
                             ("custom", 20, 1, "batch", 224),
                             ("custom", 20, 1, "group", 224),
                             ("resnet18", 20, 1, "instance", 224),
                             ("resnet18", 20, 1, "batch", 224),
                             ("resnet18", 20, 1, "group", 224),
                             ("vgg11", 20, 1, "instance", 224),
                             ("vgg11", 20, 1, "batch", 224),
                             ("vgg11", 20, 1, "group", 224),
                             ("inception_v3", 20, 1, "instance", 299),
                             ("inception_v3", 20, 1, "batch", 299),
                             ("inception_v3", 20, 1, "group", 299),
                         ])
def test_single_shapenet(feature_extractor, num_params, in_channels, norm_type,
                         img_size):

    layer_cls = HomogeneousShapeLayer
    layer_kwargs = {
        "shapes": np.random.rand(num_params + 1, 16, 2),
        "n_dims": 2,
        "use_cpp": False
    }
    net = SingleShapeNetwork(layer_cls, layer_kwargs, in_channels, norm_type,
                             img_size, feature_extractor)

    input_tensor = torch.rand(10, in_channels, img_size, img_size)

    result = net(input_tensor)["pred"]

    assert result.shape == (10, 16, 2)

    net.closure(
        model=net,
        data_dict={"data": input_tensor, "label": torch.rand(10, 16, 2)},
        optimizers={"default": DefaultOptimWrapperTorch(torch.optim.Adam(
                    net.parameters()))},
        criterions={"l1": torch.nn.L1Loss()},
        metrics={"mse": torch.nn.MSELoss()})
