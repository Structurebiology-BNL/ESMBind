import datetime
import argparse
import os
import torch
import logging
from timeit import default_timer as timer
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryRecall,
    BinaryPrecision,
)
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils import (
    logging_related,
    process_config,
    EarlyStopper,
    find_best_threshold_mcc,
    find_best_threshold_f1,
)
from data.data_process import prep_train_dataset
from torch.utils.data import DataLoader
from model.sampler import ImbalancedDatasetSampler
from model.model import ESMBindMultiModal


def main(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    logging.info(
        "Training begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    model = ESMBindMultiModal(conf.model).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.params.classifier.parameters()},
            {
                "params": model.params.encoder.parameters(),
                "lr": conf.training.encoder_learning_rate,
            },
        ],
        lr=conf.training.learning_rate,
        weight_decay=conf.training.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=15, gamma=0.7, verbose=True)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=conf.training.swa_lr)
    metric_auc = BinaryAUROC(thresholds=None)
    metric_auprc = BinaryAveragePrecision(thresholds=None)
    model.training = True  # adding Gaussian noise to embedding
    ligand_list = ["CU"]
    pos_weights = []
    train_datasets, val_datasets = [], []
    for ligand in ligand_list:
        train_dataset, val_dataset, pos_weight = prep_train_dataset(conf, ligand=ligand)
        pos_weights.append(pos_weight)
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    early_stopper = EarlyStopper(
        patience=conf.training.early_stop_patience,
        warmup_epochs=conf.training.warmup_epochs,
    )
    classification_loss = torch.nn.BCEWithLogitsLoss()
    best_threshold_f1, best_threshold_mcc = {}, {}
    best_models = []
    best_auprc = len(ligand_list) * 0.5
    for epoch in range(1, conf.training.epochs + 1):
        total_train_auprc, total_val_auprc = 0.0, 0.0
        for k, ligand in enumerate(ligand_list):
            logging.info("\nTraining for {}".format(ligand))
            # set pos_ratio to be 3 times of the actual pos_ratio (num_pos/num_neg)
            sampler = ImbalancedDatasetSampler(
                train_datasets[k], pos_ratio=3 * pos_weights[k]
            )
            dataloader_train = DataLoader(
                train_datasets[k],
                batch_size=conf.training.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=16,
                sampler=sampler,
            )
            dataloader_val = DataLoader(
                val_datasets[k],
                batch_size=conf.training.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=16,
            )
            model.ligand = ligand
            model.train()
            train_loss = 0.0
            all_outputs, all_labels = [], []
            for i, batch_data in enumerate(dataloader_train):
                feats_1, feats_2, labels = batch_data
                optimizer.zero_grad(set_to_none=True)
                feats_1 = feats_1.to(device)
                feats_2 = feats_2.to(device)
                labels = labels.to(device)
                outputs = model(feats_1, feats_2, ligand)
                loss_ = classification_loss(labels, outputs)
                loss_.backward()
                optimizer.step()

                all_outputs.append(torch.sigmoid(outputs).detach())
                all_labels.append(labels.detach())
                train_loss += loss_.detach().cpu().numpy()

            all_outputs = torch.cat(all_outputs).cpu()
            all_labels = torch.cat(all_labels).cpu().long()
            train_auc = metric_auc(all_outputs, all_labels)
            train_auprc = metric_auprc(all_outputs, all_labels)
            total_train_auprc += train_auprc
            logging.info(
                "Epoch {} train loss {:.4f}, auc {:.3f}, auprc: {:.3f}".format(
                    epoch,
                    train_loss / (i + 1),
                    train_auc,
                    train_auprc,
                )
            )
            model.eval()
            with torch.no_grad():
                all_outputs, all_labels = [], []
                for i, batch_data in enumerate(dataloader_val):
                    feats_1, feats_2, labels = batch_data
                    feats_1 = feats_1.to(device)
                    feats_2 = feats_2.to(device)
                    labels = labels.to(device)
                    outputs = model(feats_1, feats_2, ligand, training=False)
                    all_outputs.append(torch.sigmoid(outputs).detach())
                    all_labels.append(labels.detach())

                all_outputs = torch.cat(all_outputs).cpu()
                all_labels = torch.cat(all_labels).cpu().long()
                val_auc = metric_auc(all_outputs, all_labels)
                val_auprc = metric_auprc(all_outputs, all_labels)
                max_f1, best_threshold_f1[ligand] = find_best_threshold_f1(
                    all_outputs, all_labels
                )
                max_mcc, best_threshold_mcc[ligand] = find_best_threshold_mcc(
                    all_outputs, all_labels
                )
                binary_confusion_matrix = BinaryConfusionMatrix(
                    threshold=best_threshold_f1[ligand]
                )
                confusion_matrix = binary_confusion_matrix(all_outputs, all_labels)
                binary_recall = BinaryRecall(threshold=best_threshold_f1[ligand])
                binary_precision = BinaryPrecision(threshold=best_threshold_f1[ligand])
                recall = binary_recall(all_outputs, all_labels)
                precision = binary_precision(all_outputs, all_labels)
                logging.info(
                    "Epoch {} val auc {:.3f}, auprc: {:.3f}, mcc {:.3f} at threshold {:.3f}, f1 {:.3f} at threshold {:.3f}".format(
                        epoch,
                        val_auc,
                        val_auprc,
                        max_mcc,
                        best_threshold_mcc[ligand],
                        max_f1,
                        best_threshold_f1[ligand],
                    )
                )
                logging.info(
                    "val recall: {:.3f}, precision: {:.3f} at threshold {:.3f}".format(
                        recall, precision, best_threshold_f1[ligand]
                    )
                )
                logging.info(
                    "val tn: {}, fp: {}, fn: {}, tp: {}\n".format(
                        confusion_matrix[0][0],
                        confusion_matrix[0][1],
                        confusion_matrix[1][0],
                        confusion_matrix[1][1],
                    )
                )
                total_val_auprc += val_auprc
        if epoch > conf.training.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        for i, param_group in enumerate(optimizer.param_groups):
            logging.info(
                "Learning rate of parameter group {}: {:.4f}".format(
                    i, param_group["lr"]
                )
            )
        if early_stopper.early_stop(total_val_auprc, total_train_auprc):
            logging.info("Early stopping at epoch {}".format(epoch))
            break

        if (
            len(best_models) < 3
            or total_val_auprc > best_models[0][0]
            and not conf.general.debug
        ):  # save the best top-3 models
            best_auprc = total_val_auprc
            state = {
                "swa_encoder": swa_model.module.params.encoder.state_dict(),
                "swa_classifier": swa_model.module.params.classifier.state_dict(),
                "encoder": model.params.encoder.state_dict(),
                "classifier": model.params.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "swa_scheduler": swa_scheduler.state_dict(),
                "threshold_f1": best_threshold_f1,
                "threshold_mcc": best_threshold_mcc,
            }
            file_name = (
                conf.output_path
                + "/"
                + "epoch_{}".format(epoch)
                + "_auprc_{:.3f}".format(best_auprc / len(ligand_list))
            )
            torch.save(state, file_name + ".pt")
            logging.info("\n------------ Save the best model ------------\n")
            # Remove the lowest scoring model if we already have 3 models
            if len(best_models) == 3:
                _, old_file_name = best_models.pop(0)
                os.remove(old_file_name + ".pt")

            # Add the new model to the list and sort it
            best_models.append((total_val_auprc, file_name))
            best_models.sort()

    logging.info(
        "Training is done at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path of the configuration file"
    )
    args = parser.parse_args()
    conf, output_path = process_config(args.config, config_type="train")
    """
    logging related part
    """
    logging_related(output_path=output_path, debug=conf.general.debug)
    main(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
