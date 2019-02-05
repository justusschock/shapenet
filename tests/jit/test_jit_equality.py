from shapenet.layer import HomogeneousShapeLayer
from shapenet.networks import SingleShapeNetwork

from shapenet.jit import JitHomogeneousShapeLayer
from shapenet.jit import JitShapeNetwork

import torch
import numpy as np


def test_jit_equality():
    layer_kwargs = {"shapes": np.random.rand(26, 68, 2),
                    "n_dims": 2,
                    "use_cpp": False}
    net = SingleShapeNetwork(HomogeneousShapeLayer, layer_kwargs)

    jit_net = JitShapeNetwork(JitHomogeneousShapeLayer, layer_kwargs)
    jit_net.load_state_dict(net.state_dict())

    input_tensor = torch.rand(10, 1, 224, 224)

    assert (jit_net(input_tensor) - net(input_tensor)).abs().sum() < 1e-6

    traced_jit_net = torch.jit.trace(jit_net, (torch.rand(1, 1, 224, 224)))

    assert (traced_jit_net(input_tensor) -
            net(input_tensor)).abs().sum() < 1e-6
