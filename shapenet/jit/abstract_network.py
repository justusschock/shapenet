# author: Justus Schock (justus.schock@rwth-aachen.de)

import torch
from abc import abstractmethod


class AbstractShapeNetwork(torch.jit.ScriptModule):
    """
    Abstract JIT Network

    """

    def __init__(self, **kwargs):

        super().__init__(optimize=True)

    @staticmethod
    def norm_type_to_class(norm_type):
        norm_dict = {'instance': torch.nn.InstanceNorm2d,
                     'batch': torch.nn.BatchNorm2d}

        norm_class = norm_dict.get(norm_type, None)

        return norm_class


class AbstractFeatureExtractor(torch.jit.ScriptModule):
    """
    Abstract Feature Extractor Class all further feature extractors
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

    @torch.jit.script_method
    def forward(self, input_batch):
        """
        Feed batch through network

        Parameters
        ----------
        input_batch : :class:`torch.Tensor`
            batch to feed through network

        Returns
        -------
        :class:`torch.Tensor`
            extracted features

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
        :class:`torch.jit.ScriptModule`
            ensembled model

        """
        raise NotImplementedError
