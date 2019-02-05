from shapenet.networks.feature_extractors import \
    Img224x224Kernel7x7SeparatedDims
from shapenet.networks.utils import CustomGroupNorm
import torch
import pytest


@pytest.mark.parametrize("num_outputs,num_in_channels,norm_class,p_dropout",
                         [
                             (16, 1, torch.nn.InstanceNorm2d, 0.1),
                             (16, 1, torch.nn.BatchNorm2d, 0.5),
                             (16, 1, CustomGroupNorm, 0.5),
                             (75, 125, torch.nn.InstanceNorm2d, 0.),
                             (75, 125, torch.nn.BatchNorm2d, 0.),
                             (75, 125, CustomGroupNorm, 0.)
                         ])
def test_224_img_size_7_kernel_size_separated_dims(num_outputs, num_in_channels,
                                                   norm_class, p_dropout):
    net = Img224x224Kernel7x7SeparatedDims(num_in_channels, num_outputs,
                                           norm_class, p_dropout)

    input_tensor = torch.rand(16, num_in_channels, 224, 224)

    assert net(input_tensor).shape == (16, num_outputs, 1, 1)
