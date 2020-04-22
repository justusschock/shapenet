# author: Justus Schock (justus.schock@rwth-aachen.de)


def train_shapenet():
    """
    Trains a single shapenet with config file from comandline arguments

    See Also
    --------
    :class:`delira.training.PyTorchNetworkTrainer`
    
    """

    import logging
    import numpy as np
    import torch
    from shapedata.single_shape import SingleShapeDataset
    from delira.training import PyTorchNetworkTrainer
    from ..utils import Config
    from ..layer import HomogeneousShapeLayer
    from ..networks import SingleShapeNetwork
    from delira.logging import TrixiHandler
    from trixi.logger import PytorchVisdomLogger
    from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch
    from delira.data_loading import BaseDataManager, RandomSampler, \
        SequentialSampler
    import os
    import argparse
    from sklearn.metrics import mean_squared_error
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    config = Config()

    config_dict = config(os.path.abspath(args.config))

    shapes = np.load(os.path.abspath(config_dict["layer"].pop("pca_path"))
                     )["shapes"][:config_dict["layer"].pop("num_shape_params") + 1]

# layer_cls = HomogeneousShapeLayer

    net = SingleShapeNetwork(
        HomogeneousShapeLayer, {"shapes": shapes,
                                **config_dict["layer"]},
        img_size=config_dict["data"]["img_size"],
        **config_dict["network"])

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    if args.verbose:
        print("Number of Parameters: %d" % num_params)

    criterions = {"L1": torch.nn.L1Loss()}
    metrics = {"MSE": torch.nn.MSELoss()}
    
    def numpy_mse(pred, target):
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        
        return mean_square_error(target, pred)

    mixed_prec = config_dict["training"].pop("mixed_prec", False)

    config_dict["training"]["save_path"] = os.path.abspath(
        config_dict["training"]["save_path"])

    trainer = PyTorchNetworkTrainer(
        net, losses=criterions, train_metrics=metrics,
        val_metrics={"MSE": numpy_mse},
        lr_scheduler_cls=ReduceLROnPlateauCallbackPyTorch,
        lr_scheduler_params=config_dict["scheduler"],
        optimizer_cls=torch.optim.Adam,
        optimizer_params=config_dict["optimizer"],
        mixed_precision=mixed_prec,
        key_mapping={"input_images": "data"},
        **config_dict["training"])

    if args.verbose:
        print(trainer.input_device)

        print("Load Data")
    dset_train = SingleShapeDataset(
        os.path.abspath(config_dict["data"]["train_path"]),
        config_dict["data"]["img_size"], config_dict["data"]["crop"],
        config_dict["data"]["landmark_extension_train"],
        cached=config_dict["data"]["cached"],
        rotate=config_dict["data"]["rotate_train"],
        random_offset=config_dict["data"]["offset_train"]
    )

    if config_dict["data"]["test_path"]:
        dset_val = SingleShapeDataset(
            os.path.abspath(config_dict["data"]["test_path"]),
            config_dict["data"]["img_size"], config_dict["data"]["crop"],
            config_dict["data"]["landmark_extension_test"],
            cached=config_dict["data"]["cached"],
            rotate=config_dict["data"]["rotate_test"],
            random_offset=config_dict["data"]["offset_test"]
        )

    else:
        dset_val = None

    mgr_train = BaseDataManager(
        dset_train,
        batch_size=config_dict["data"]["batch_size"],
        n_process_augmentation=config_dict["data"]["num_workers"],
        transforms=None,
        sampler_cls=RandomSampler
    )
    mgr_val = BaseDataManager(
        dset_val,
        batch_size=config_dict["data"]["batch_size"],
        n_process_augmentation=config_dict["data"]["num_workers"],
        transforms=None,
        sampler_cls=SequentialSampler
    )

    if args.verbose:
        print("Data loaded")
    if config_dict["logging"].pop("enable", False):
        logger_cls = PytorchVisdomLogger

        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                TrixiHandler(
                                    logger_cls, **config_dict["logging"])
                            ])

    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.NullHandler()])

    logger = logging.getLogger("Test Logger")
    logger.info("Start Training")

    if args.verbose:
        print("Start Training")

    trainer.train(config_dict["training"]["num_epochs"], mgr_train, mgr_val,
                  config_dict["training"]["val_score_key"],
                  val_score_mode='lowest')


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train_shapenet()
