# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
from abc import abstractmethod
from delira.models import AbstractPyTorchNetwork
from .utils import CustomGroupNorm


class AbstractShapeNetwork(AbstractPyTorchNetwork):
    """
    Abstract base Class to provide a convenient norm_class_mapping

    """
    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :
            keyword arguments (forwarded to parent class)
        """
        super().__init__(**kwargs)

    @staticmethod
    def norm_type_to_class(norm_type):
        """
        helper function to map a string to an actual normalization class

        Parameters
        ----------
        norm_type : str
            string specifying the normalization class

        Returns
        -------
        type
            Normalization Class (subclass of torch.nn.Module)

        """
        norm_dict = {'instance': torch.nn.InstanceNorm2d,
                     'batch': torch.nn.BatchNorm2d,
                     'group': CustomGroupNorm}

        norm_class = norm_dict.get(norm_type, None)

        return norm_class


class AbstractFeatureExtractor(torch.nn.Module):
    """
    Abstract Feature Extractor Class all further feature extracotrs
    should be derived from

    """
    def __init__(self, in_channels, out_params, norm_class, p_dropout=0):
        """

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_params : int
            number of outputs
        norm_class : Any
            Class implementing a normalization
        p_dropout : float
            dropout probability

        """
        super().__init__()
        self.model = self._build_model(in_channels, out_params, norm_class,
                                       p_dropout)

    def forward(self, input_batch):
        """
        Feed batch through network

        Parameters
        ----------
        input_batch : torch.Tensor
            batch to feed through network

        Returns
        -------
        torch.Tensor
            exracted features

        """
        return self.model(input_batch)

    @staticmethod
    @abstractmethod
    def _build_model(in_channels, out_features, norm_class, p_dropout):
        """
        Build the actual model structure

        Parameters
        ----------
        in_channels : int
            number of input channels
        out_features : int
            number of outputs
        norm_class : Any
            class implementing a normalization
        p_dropout : float
            dropout probability

        Returns
        -------
        torch.nn.Module
            ensembled model
        """
        raise NotImplementedError
