import numpy as np
import logging
import sys
import json
import torch
import datetime
from pathlib import Path
from ml_collections import config_dict
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
)
from model.model import ESMBindMultiModal, ESMBindSingle


def find_best_threshold_mcc(outputs, labels):
    max_mcc = -1  # Initialize to -1
    best_threshold = 0
    for threshold in np.arange(0.01, 1.01, 0.01):
        binary_mcc = BinaryMatthewsCorrCoef(threshold=threshold)
        mcc = binary_mcc(outputs, labels)
        if mcc > max_mcc:
            max_mcc = mcc
            best_threshold = threshold
    return max_mcc, best_threshold


def find_best_threshold_f1(outputs, labels):
    best_threshold = 0
    best_f1 = -1  # Initialize to -1
    for threshold in np.arange(0.01, 1.01, 0.01):  # step of 0.01
        binary_f1 = BinaryF1Score(threshold=threshold)
        f1 = binary_f1(outputs, labels)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold


class EarlyStopper:
    """
    A class for implementing early stopping to prevent overfitting during model training.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        warmup_epochs (int): Number of initial epochs to ignore during early stopping.
        counter (int): Counts the number of epochs without improvement.
        gap_counter (int): Counts the number of consecutive epochs where the gap between
                           training and validation performance increases.
        prev_gap (float): The gap between training and validation performance in the previous epoch.
        best_auprc (float): The best validation AUPRC observed.
        epoch (int): Current epoch number.

    Methods:
        early_stop(validation_auprc, training_auprc): Determines whether to stop training based on
                                                      validation performance and the gap between training
                                                      and validation performance.
    """

    def __init__(self, patience=6, min_delta=1e-7, warmup_epochs=10):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.gap_counter = 0  # Counter for consecutive increases in gap
        self.prev_gap = 0  # Previous gap between training and validation AUPRC
        self.best_auprc = 0  # Best validation AUPRC
        self.epoch = 0  # To keep track of the number of epochs

    def early_stop(self, validation_auprc, training_auprc):
        self.epoch += 1  # Increment epoch count
        stop_training = False  # Flag to determine if training should stop

        # Calculate the gap
        gap = training_auprc - validation_auprc

        # Update best AUPRC
        if validation_auprc > self.best_auprc:
            self.best_auprc = validation_auprc
            self.counter = 0  # Reset counter
        elif validation_auprc < (self.best_auprc + self.min_delta):
            self.counter += 1  # Increment counter if no improvement in validation AUPRC

        # Check the gap if warm-up is complete
        if self.epoch > self.warmup_epochs:
            if gap > 0:
                if gap > self.prev_gap:
                    self.gap_counter += (
                        1  # Increment gap counter if the gap has increased
                    )
                else:
                    self.gap_counter = (
                        0  # Reset gap counter if the gap has decreased or is negative
                    )
            else:
                self.gap_counter = 0  # Reset gap counter if the gap is negative

            # Check if gap has consecutively increased
            if self.gap_counter >= self.patience:
                logging.info(
                    f"Early stopping because performance gap increased consecutively for {self.patience} epochs."
                )
                stop_training = True

        # Update the previous gap
        self.prev_gap = gap

        # Check if patience has run out
        if self.counter >= self.patience:
            logging.info(
                f"Early stopping because validation AUPRC did not improve for {self.patience} epochs."
            )
            stop_training = True

        return stop_training


def load_ensemble_model(conf, device, multi_modal=True):
    models = []
    f1_threshold_list, mcc_threshold_list = [], []
    if "inference" in conf:
        ensemble_path = conf.inference.ensemble_path
    elif "test" in conf:
        ensemble_path = conf.test.ensemble_path
    for i in range(1, 6):
        if multi_modal:
            model = ESMBindMultiModal(conf.model)
        else:
            if conf.data.feature_1 == "esm":
                conf.model.feature_dim = 1280
            elif conf.data.feature_1 == "esm_if":
                conf.model.feature_dim = 512
            model = ESMBindSingle(conf.model)
        checkpoint = torch.load(ensemble_path + f"/fold_{i}.pt", map_location="cpu")
        logging.info("load weights for fold {} from {}".format(i, ensemble_path))
        model.params.encoder.load_state_dict(checkpoint["swa_encoder"], strict=False)
        model.params.classifier.load_state_dict(
            checkpoint["swa_classifier"], strict=False
        )
        f1_threshold_list.append(checkpoint["threshold_f1"])
        mcc_threshold_list.append(checkpoint["threshold_mcc"])
        model.training = False
        model.eval()
        model.to(device)
        models.append(model)
    threshold_f1, threshold_mcc = {}, {}
    keys = f1_threshold_list[0].keys()
    # Calculate average for each key
    for key in keys:
        threshold_f1[key] = sum(d[key] for d in f1_threshold_list) / len(
            f1_threshold_list
        )
        threshold_mcc[key] = sum(d[key] for d in mcc_threshold_list) / len(
            mcc_threshold_list
        )

    return models, threshold_f1, threshold_mcc


def load_json(file_path):
    """Load and return a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
        raise
    except FileNotFoundError:
        logging.error(f"Config file not found: {file_path}")
        raise


def setup_output_path(conf, config_type):
    """Setup the output directory and return the path."""
    if conf.get("general", {}).get("debug", False):
        return None

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_path = Path("multi_modal_binding/results") / config_type / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    conf["output_path"] = str(output_path)
    with open(output_path / "config.json", "w") as f:
        json.dump(conf, f, indent=4)

    return output_path


def process_config(conf_path, config_type="train"):
    """
    Process the configuration for training from scratch or inference using ensemble.

    Parameters:
    - conf_path: the path to the configuration file.
    - config_type: The mode of operation, either 'train' or 'inference'.

    Returns:
    - A tuple of the updated configuration dictionary and the output path as a Path object (if created).
    """
    if config_type not in ["train", "inference"]:
        raise ValueError("config_type must be either 'train' or 'inference'")

    conf = load_json(conf_path)

    if config_type == "inference":
        ensemble_path = conf.get("inference", {}).get("ensemble_path")
        if not ensemble_path:
            raise ValueError("ensemble_path must be specified for inference mode")

        ensemble_config_path = Path(ensemble_path) / "config.json"
        ensemble_conf = load_json(ensemble_config_path)
        ensemble_conf.pop("general", None)  # Remove general settings to avoid conflicts
        conf.update(ensemble_conf)  # Combine with main config

    output_path = setup_output_path(conf, config_type)

    return config_dict.ConfigDict(conf), output_path


def logging_related(output_path=None, debug=True, training=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if training:
        log_filename = str(output_path) + "/training.log"
    else:
        log_filename = str(output_path) + "/inference.log"
    if not debug:
        assert output_path is not None, "need valid log output path"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info("Output path: {}".format(output_path))
