# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
from .abstract_network import AbstractFeatureExtractor


class Conv2dRelu(torch.jit.ScriptModule):
    """
    Block holding one Conv2d and one ReLU layer
    
    """

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        **args :
            positional arguments (passed to Conv2d)
        **kwargs : dict
            keyword arguments (passed to Conv2d)

        """
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._relu = torch.nn.ReLU()

    @torch.jit.script_method
    def forward(self, input_batch):
        """
        Forward batch though layers

        Parameters
        ----------
        input_batch : class:`torch.Tensor`
            input batch

        Returns
        -------
        class:`torch.Tensor`
            result

        """
        return self._relu(self._conv(input_batch))


class Img224x224Kernel7x7SeparatedDims(AbstractFeatureExtractor):
    @staticmethod
    def _build_model(in_channels, out_params, norm_class, p_dropout):
        """
        Build the actual model structure

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_params : int
            number of outputs
        norm_class : Any
            class implementing a normalization
        p_dropout : float
            dropout probability

        Returns
        -------
        :class:`torch.jit.ScriptModule`
            ensembled model

        """
        model = torch.nn.Sequential()

        model.add_module("conv_1", Conv2dRelu(in_channels, 64, (7, 1)))
        model.add_module("conv_2", Conv2dRelu(64, 64, (1, 7)))

        model.add_module("down_conv_1", Conv2dRelu(64, 128, (7, 7), stride=2))
        if norm_class is not None:
            model.add_module("norm_1", norm_class(128))
        if p_dropout:
            model.add_module("dropout_1", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_3", Conv2dRelu(128, 128, (7, 1)))
        model.add_module("conv_4", Conv2dRelu(128, 128, (1, 7)))

        model.add_module("down_conv_2", Conv2dRelu(128, 256, (7, 7), stride=2))
        if norm_class is not None:
            model.add_module("norm_2", norm_class(256))
        if p_dropout:
            model.add_module("dropout_2", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_5", Conv2dRelu(256, 256, (5, 1)))
        model.add_module("conv_6", Conv2dRelu(256, 256, (1, 5)))

        model.add_module("down_conv_3", Conv2dRelu(256, 256, (5, 5), stride=2))
        if norm_class is not None:
            model.add_module("norm_3", norm_class(256))
        if p_dropout:
            model.add_module("dropout_3", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_7", Conv2dRelu(256, 256, (5, 1)))
        model.add_module("conv_8", Conv2dRelu(256, 256, (1, 5)))

        model.add_module("down_conv_4", Conv2dRelu(256, 128, (5, 5), stride=2))
        if norm_class is not None:
            model.add_module("norm_4", norm_class(128))
        if p_dropout:
            model.add_module("dropout_4", torch.nn.Dropout2d(p_dropout))

        model.add_module("conv_9", Conv2dRelu(128, 128, (3, 1)))
        model.add_module("conv_10", Conv2dRelu(128, 128, (1, 3)))
        model.add_module("conv_11", Conv2dRelu(128, 128, (3, 1)))
        model.add_module("conv_12", Conv2dRelu(128, 128, (1, 3)))

        model.add_module("final_conv", torch.nn.Conv2d(128, out_params,
                                                       (2, 2)))

        return torch.jit.trace(model, torch.rand(5, in_channels, 224, 224))
