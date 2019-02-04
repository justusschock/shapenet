from ..jit.homogeneous_shape_layer import HomogeneousShapeLayer
from ..jit.shape_network import ShapeNetwork
import torch
import argparse
import numpy as np
import os


def create_jit_net_from_config_and_weight(config_dict, weight_file):
    shapes = np.load(os.path.abspath(
        config_dict["layer"].pop("pca_path"))
    )["shapes"][:config_dict["layer"].pop("num_shape_params") + 1]

    net = ShapeNetwork(HomogeneousShapeLayer, {
        "shapes": shapes,
        "n_dims": config_dict["layer"]["n_dims"],
        "use_cpp": False})

    input_tensor = torch.rand(1, config_dict["network"]["in_channels"],
                              config_dict["data"]["img_size"],
                              config_dict["data"]["img_size"]
                              )

    try:
        net.load_state_dict(
            torch.load(weight_file,
                       map_location="cpu")["state_dict"]["model"]
        )
    except:
        net.load_state_dict(
            torch.load(weight_file, map_location="cpu")
        )

    traced = torch.jit.trace(net, (input_tensor))

    return traced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="Configuration")
    parser.add_argument("-w", "--weight_file", type=str, help="Weights")
    parser.add_argument("-o", "--out_file", type=str, help="Outputfile")

    args = parser.parse_args()

    import os
    os.makedirs(os.path.split(args.out_file)[0], exist_ok=True)

    from shapenet.utils import Config
    traced = create_jit_net_from_config_and_weight(Config()(
        os.path.abspath(args.config_file)),
        os.path.abspath(args.weight_file))

    traced.save(os.path.abspath(args.out_file))


if __name__ == '__main__':
    main()
