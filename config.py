import torch


def initialize_default_config(dataset):
    if dataset == "Mayo":
        default_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "dataset": "Mayo",
            "angular_range": 180,
            "n_angles": 60,
            "geometry": "fanflat",
            "image_shape": [1, 256, 256],
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
            "image_shape": [1, 256, 256],
        }
        _, nx, ny = default_config["image_shape"]
        default_config["det_size"] = int(max(nx, ny) * 2)

    else:
        raise NotImplementedError

    return default_config


def parse_config(config):
    import argparse

    # Build parser dict
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--image_shape",
        nargs="+",
        required=False,
        type=int,
        default=None,
        help="Image shape. Represented as a sequence of three numbers, i.e. n_ch, nx, ny.",
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        default=None,
        help="Device to use. Choose between cuda or cpu.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=False,
        type=str,
        default=None,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--geometry",
        required=False,
        type=str,
        default=None,
        help="Projection geometry.",
    )
    parser.add_argument(
        "--n_angles",
        required=False,
        type=int,
        default=None,
        help="Number of projection angles.",
    )
    parser.add_argument(
        "--det_size",
        required=False,
        type=int,
        default=None,
        help="Number of detector pixels.",
    )
    parser.add_argument(
        "--angular_range",
        required=False,
        type=int,
        default=None,
        help="Maximum acquisition angle. Resulting projections in range [0, angular_range]. Measured in degrees.",
    )
    args = dict(vars(parser.parse_args()))

    # Remove keys with None values from the args dict
    args = {k: v for k, v in args.items() if v is not None}

    # Merge with config dict
    merged_dict = config.copy()

    # Update the merged_dict with dict2 values
    merged_dict.update(args)
    print(merged_dict)
    return merged_dict
