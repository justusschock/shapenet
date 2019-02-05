from shapenet.utils import Config
import os

def test_config():
    config = Config()(os.path.join(os.path.split(os.path.abspath(__file__))[0],
                        "dummy.config"))
    
    target_dict = {
        "network":
        {
            "in_channels": 1,
            "norm_type": 'instance',
            "feature_extractor": False
        },
        "layer":
        {
            "pca_path": "test123",
            "num_shape_params": 25,
            "n_dims": 2,
            "use_cpp": False
        },
        "optimizer":
        {
            "lr": 0.0001
        },
        "scheduler":
        {
            "factor": 0.1,
            "patience": 5,
            "cooldown": 0
        },
        "training":
        {
            "save_path": "test234",
            "gpu_ids": [0],
            "save_freq": 1,
            "num_epochs": 200,
            "val_score_key": "val_MSE"
        },
        "data":
        {
            "train_path": "test345",
            "test_path": "test456",
            "crop": 0.1,
            "landmark_extension_train": ".pts",
            "landmark_extension_test": ".pts",
            "batch_size": 1,
            "cached": False,
            "num_workers": 1,
            "img_size": 224,
            "rotate_train": 90,
            "rotate_test": 45,
            "offset_train": 30,
            "offset_test": 20
        },
        "logging":
        {
            "enable": False,
            "port": 9999,
            "name": "TEST",
            "server": "http://localhost"
        }
    }

    assert config == target_dict
