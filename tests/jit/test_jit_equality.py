from shapenet.jit import JitHomogeneousShapeLayer
from shapenet.jit import JitShapeNetwork

import torch
import numpy as np


def test_jit_equality():
    layer_kwargs = {"shapes": np.random.rand(26, 68, 2),
                    "n_dims": 2,
                    "use_cpp": False}

    input_tensor = torch.rand(10, 1, 224, 224)

    jit_net = JitShapeNetwork(JitHomogeneousShapeLayer, layer_kwargs)

    assert torch.jit.trace(jit_net, (torch.rand(1, 1, 224, 224)))
