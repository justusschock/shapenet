
def predict():
    import numpy as np
    import torch
    from tqdm import tqdm
    import os
    from matplotlib import pyplot as plt
    from shapenet.utils import Config
    from shapenet.layer import HomogeneousShapeLayer
    from shapenet.networks import SingleShapeNetwork
    from shapedata.single_shape import SingleShapeDataProcessing, \
                                        SingleShapeSingleImage2D
    from shapedata.io import pts_exporter
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="If Flag is specified, results will be plotted")
    parser.add_argument("-d", "--in_path", type=str, help="Input Data Dir")
    parser.add_argument("-s", "--out_path", default="./outputs", type=str,
                        help="Output Data Dir")
    parser.add_argument("-w", "--weight_file", type=str, help="Model Weights")
    parser.add_argument("-c", "--config_file", type=str, help="Configuration")

    args = parser.parse_args()
    config = Config()
    config_dict = config(os.path.abspath(args.config_file))

    try:
        net = torch.jit.load(os.path.abspath(args.weight_file))
        net.eval()
        net.cpu()

    except RuntimeError:
        net_layer = HomogeneousShapeLayer

        if config_dict["training"].pop("mixed_prec", False):
            try:
                from apex import amp
                amp.init()
            except:
                pass

        shapes = np.load(os.path.abspath(config_dict["layer"].pop("pca_path"))
                        )["shapes"][:config_dict["layer"].pop("num_shape_params") + 1]

        net = SingleShapeNetwork(
            net_layer, {"shapes": shapes,
                        **config_dict["layer"]},
            img_size=config_dict["data"]["img_size"],
            **config_dict["network"])

        try:
            net.load_state_dict(torch.load(
                os.path.abspath(args.weight_file))["state_dict"]["model"])
        except KeyError:
            net.load_state_dict(torch.load(os.path.abspath(args.weight_file)))

        net = net.to("cpu")
        net = net.eval()

    data = SingleShapeDataProcessing._get_files(
        os.path.abspath(args.in_path), extensions=[".png", ".jpg"])

    def process_sample(sample, img_size, net, device, crop=0.1):
        lmk_bounds = sample.get_landmark_bounds(sample.lmk)
        min_y, min_x, max_y, max_x = lmk_bounds
        range_x = max_x - min_x
        range_y = max_y - min_y

        max_range = max(range_x, range_y) * (1 + crop)

        center_x = min_x + range_x / 2
        center_y = min_y + range_y / 2

        tmp = sample.crop(center_y - max_range / 2,
                        center_x - max_range / 2,
                        center_y + max_range / 2,
                        center_x + max_range / 2)

        img_tensor = torch.from_numpy(
            tmp.to_grayscale().resize((img_size, img_size)).img.transpose(2, 0,
                                                                          1)
        ).to(torch.float).unsqueeze(0).to(device)

        pred = net(img_tensor).cpu().numpy()[0]
        
        pred = pred * np.array([max_range / img_size, max_range / img_size])
        pred = pred + np.asarray([center_y - max_range / 2,
                                center_x - max_range / 2])

        return pred

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():

        if torch.cuda.is_available():
            net = net.cuda()

        if args.visualize:
            pred_path = os.path.join(os.path.abspath(args.out_path), "pred")
            vis_path = os.path.join(os.path.abspath(args.out_path),
                                    "visualization")
            os.makedirs(vis_path, exist_ok=True)
        else:
            pred_path = os.path.abspath(args.out_path)

        os.makedirs(pred_path, exist_ok=True)

        for idx, file in enumerate(tqdm(data)):

            _data = SingleShapeSingleImage2D.from_files(file)

            pred = process_sample(_data, img_size=net.img_size, net=net,
                                  device=device)

            fname = os.path.split(_data.img_file)[-1].rsplit(".", 1)[0]

            if args.visualize:
                view_kwargs = {}
                if _data.is_gray:
                    view_kwargs["cmap"] = "gray"
                fig = _data.view(True, **view_kwargs)
                plt.gca().scatter(pred[:, 1], pred[:, 0], s=5, c="C1")
                plt.gca().legend(["GT", "Pred"])
                plt.gcf().savefig(os.path.join(vis_path, fname + ".png"))
                plt.close()

            _data.save(pred_path, fname, "PTS")
            pts_exporter(pred, os.path.join(pred_path, fname + "_pred.pts"))

if __name__ == '__main__':
    predict()