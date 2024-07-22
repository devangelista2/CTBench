import torch


def initialize_default_config(dataset):
    if dataset == "Mayo":
        default_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dataset": "Mayo",
            "angular_range": 180,
            "n_angles": 60,
            "geometry": "fanflat",
            "image_shape": (1, 256, 256),
        }
        _, nx, ny = default_config["image_shape"]
        default_config["det_size"] = int(max(nx, ny) * 2)

    elif dataset == "COULE":
        default_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dataset": "COULE",
            "angular_range": 180,
            "n_angles": 60,
            "geometry": "fanflat",
            "image_shape": (1, 256, 256),
        }
        _, nx, ny = default_config["image_shape"]
        default_config["det_size"] = int(max(nx, ny) * 2)

    else:
        raise NotImplementedError

    return default_config
