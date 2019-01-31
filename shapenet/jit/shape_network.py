# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
import torchvision.models
import logging

from .feature_extractors import Img224x224Kernel7x7SeparatedDims
from .abstract_network import AbstractShapeNetwork

logger = logging.getLogger(__file__)


class ShapeNetwork(AbstractShapeNetwork):
    """
    Network to Predict a single shape

    """

    __constants__ = ['num_out_params']

    def __init__(self, layer_cls,
                 layer_kwargs,
                 in_channels=1,
                 norm_type='instance',
                 img_size=224,
                 tiny=False,
                 feature_extractor=None,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        layer_cls : type, subclass of ``torch.nn.Module``
            Class to instantiate the last layer (usually a shape-constrained
            or transformation layer)
        layer_kwargs : dict
            keyword arguments to create an instance of `layer_cls`
        in_channels : int
            number of input channels
        norm_type : str or None
            Indicates the type of normalization used in this network;
            Must be one of [None, 'instance', 'batch', 'group']
        **kwargs :
            additional keyword arguments

        """

        super().__init__()

        self._kwargs = kwargs

        self._out_layer = layer_cls(**layer_kwargs)
        self.num_out_params = self._out_layer.num_params
        self.img_size = img_size
        norm_class = self.norm_type_to_class(norm_type)

        args = [in_channels, self.num_out_params, norm_class]
        feature_kwargs = {}

        if img_size == 224:
            if feature_extractor and hasattr(torchvision.models,
                                             feature_extractor):
                feature_extractor_cls = getattr(torchvision.models,
                                                feature_extractor)
                args = [False]
                feature_kwargs = {"num_classes": self.num_out_params}

            else:
                feature_extractor_cls = Img224x224Kernel7x7SeparatedDims

        else:
            raise ValueError("No known dimension for image size found")
        # self._model = Img224x224Kernel7x7SeparatedDims(
        #     in_channels, self._out_layer.num_params, norm_class
        # )

        model = feature_extractor_cls(*args, **feature_kwargs)

        if isinstance(model, torchvision.models.VGG):
            model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                *list(model.features.children())[1:]
            )

        elif isinstance(model, torchvision.models.ResNet):
            model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                          stride=2, padding=3,
                                          bias=False)

        elif isinstance(model, torchvision.models.Inception3):
            model.Conv2d_1a_3x3 = \
                torchvision.models.inception.BasicConv2d(in_channels, 32,
                                                         kernel_size=3,
                                                         stride=2)

        elif isinstance(model, torchvision.models.DenseNet):
            out_channels = list(model.features.children()
                                )[0].out_channels
            model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=7,
                                stride=2, padding=3, bias=False),
                *list(model.features.children())[1:]
            )

        elif isinstance(model, torchvision.models.SqueezeNet):
            out_channels = list(model.features.children()
                                )[0].out_channels
            model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=7,
                                stride=2),
                *list(model.features.children())[1:]
            )

        elif isinstance(model, torchvision.models.AlexNet):
            out_channels = list(model.features.children()
                                )[0].out_channels
            model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=11,
                                stride=4, padding=2),
                *list(model.features.children())[1:]
            )

        self._model = torch.jit.trace(model,
                                      torch.rand(10, in_channels,
                                                 img_size, img_size))

    @torch.jit.script_method
    def forward(self, input_images):
        """
        Forward input batch through network and shape layer

        Parameters
        ----------
        input_images : torch.Tensor
            input batch

        Returns
        -------
        torch.Tensor
            predicted shapes

        """

        features = self._model(input_images)

        return self._out_layer(features.view(input_images.size(0),
                                             self.num_out_params, 1, 1))

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module):
        if isinstance(model, torch.nn.Module):
            self._model = model
        else:
            raise AttributeError("Invalid Model")
