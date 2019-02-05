# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
import torchvision.models
import logging

from ..feature_extractors import Img224x224Kernel7x7SeparatedDims
from ..abstract_network import AbstractShapeNetwork

logger = logging.getLogger(__file__)


class ShapeNetwork(AbstractShapeNetwork):
    """
    Network to Predict a single shape
    """

    def __init__(self, layer_cls,
                 layer_kwargs,
                 in_channels=1,
                 norm_type='instance',
                 img_size=224,
                 feature_extractor=None,
                 **kwargs
                 ):
        """

        Parameters
        ----------
        layer_cls :
            Class to instantiate the last layer (usually a shape-constrained
            or transformation layer)
        layer_kwargs : dict
            keyword arguments to create an instance of `layer_cls`
        in_channels : int
            number of input channels
        norm_type : string or None
            Indicates the type of normalization used in this network;
            Must be one of [None, 'instance', 'batch', 'group']
        kwargs :
            additional keyword arguments

        """

        super().__init__(layer_cls=layer_cls,
                         layer_kwargs=layer_kwargs,
                         in_channels=in_channels,
                         norm_type=norm_type,
                         img_size=img_size,
                         feature_extractor=feature_extractor,
                         **kwargs)
        self._kwargs = kwargs

        self._model = None
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

        elif img_size == 299 and feature_extractor == "inception_v3":
            feature_extractor_cls = torchvision.models.inception_v3
            args = [False]
            feature_kwargs = {"num_classes": self.num_out_params,
                              "aux_logits": False}

        else:
            raise ValueError("No known dimension for image size found")
        # self._model = Img224x224Kernel7x7SeparatedDims(
        #     in_channels, self._out_layer.num_params, norm_class
        # )

        self._model = feature_extractor_cls(*args, **feature_kwargs)

        if isinstance(self._model, torchvision.models.VGG):
            self._model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                *list(self._model.features.children())[1:]
            )

        elif isinstance(self._model, torchvision.models.ResNet):
            self._model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                                stride=2, padding=3,
                                                bias=False)

        elif isinstance(self._model, torchvision.models.Inception3):
            self._model.Conv2d_1a_3x3 = \
                torchvision.models.inception.BasicConv2d(in_channels, 32,
                                                         kernel_size=3,
                                                         stride=2)

        elif isinstance(self._model, torchvision.models.DenseNet):
            out_channels = list(self._model.features.children()
                                )[0].out_channels
            self._model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=7,
                                stride=2, padding=3, bias=False),
                *list(self._model.features.children())[1:]
            )

        elif isinstance(self._model, torchvision.models.SqueezeNet):
            out_channels = list(self._model.features.children()
                                )[0].out_channels
            self._model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=7,
                                stride=2),
                *list(self._model.features.children())[1:]
            )

        elif isinstance(self._model, torchvision.models.AlexNet):
            out_channels = list(self._model.features.children()
                                )[0].out_channels
            self._model.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=11,
                                stride=4, padding=2),
                *list(self._model.features.children())[1:]
            )

    def forward(self, input_images):
        """
        Forward input batch through network and shape layer

        Parameters
        ----------
        input_images : :class:`torch.Tensor`
            input batch

        Returns
        -------
        :class:`torch.Tensor`
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

    @staticmethod
    def closure(model, data_dict: dict,
                optimizers: dict, criterions={}, metrics={},
                fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model : :class:`ShapeNetwork`
            trainable model
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary of optimizers to optimize model's parameters
        criterions : dict
            dict holding the criterions to calculate errors
            (gradients from different criterions will be accumulated)
        metrics : dict
            dict holding the metrics to calculate
        fold : int
            Current Fold in Crossvalidation (default: 0)
        **kwargs :
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict criterions)
        list
            Arbitrary number of predictions as :class:`torch.Tensor`

        Raises
        ------
        AssertionError
            if optimizers or criterions are empty or the optimizers are not
            specified
        """

        assert (optimizers and criterions) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad

        else:
            context_man = torch.no_grad

        with context_man():

            inputs = data_dict.pop("data")
            preds = model(inputs)

            if data_dict:

                for key, crit_fn in criterions.items():
                    _loss_val = crit_fn(preds, *data_dict.values())
                    loss_vals[key] = _loss_val.detach()
                    total_loss += _loss_val

                with torch.no_grad():
                    for key, metric_fn in metrics.items():
                        metric_vals[key] = metric_fn(
                            preds, *data_dict.values())

        if optimizers:
            optimizers['default'].zero_grad()
            total_loss.backward()
            optimizers['default'].step()

        else:

            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"value": val.item(), "name": key,
                                    "env_appendix": "_%02d" % fold
                                    }})

        return metric_vals, loss_vals, [preds]
