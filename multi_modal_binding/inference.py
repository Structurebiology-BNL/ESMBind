import datetime
import argparse
import torch
import logging
import pickle
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from utils import logging_related, process_config, load_ensemble_model
from data.data_process import prep_test_dataset


def main(conf):
    RANDOM_SEED = int(conf.general.seed)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = (
        torch.device("cuda:{}".format(conf.general.gpu_id))
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.cuda.empty_cache()
    logging.info(
        "Test begins at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )
    # load 5-fold models
    models, threshold_f1, threshold_mcc = load_ensemble_model(conf, device)
    ligand_list = [
        "MG",
        "FE",
        "CU",
        "CO",
        "CA",
        "MN",
        "ZN",
    ]
    predictions = {}
    predictions["threshold_f1"] = threshold_f1
    predictions["threshold_mcc"] = threshold_mcc
    test_dataset = prep_test_dataset(conf)
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
        collate_fn=test_dataset.collate_fn,
    )
    for ligand in ligand_list:
        predictions[ligand] = {}
        with torch.no_grad():
            for batch_data in dataloader_test:
                id, feats_1, feats_2, masks = batch_data
                feats_1 = feats_1.to(device)
                feats_2 = feats_2.to(device)
                masks = masks.to(device)
                outputs = []
                for model in models:
                    model.ligand = ligand
                    output = model(feats_1, feats_2, ligand)
                    output = torch.sigmoid(torch.masked_select(output, masks.bool()))
                    outputs.append(output)
                outputs = torch.stack(outputs).mean(
                    0
                )  # average the predictions from 5 models
                predictions[ligand][id[0]] = outputs.detach().cpu().numpy()

    with open(conf.output_path + "/predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    logging.info(
        "Test is done at {}".format(datetime.datetime.now().strftime("%m-%d %H:%M"))
    )


if __name__ == "__main__":
    start = timer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path of the configuration file"
    )
    args = parser.parse_args()
    conf, output_path = process_config(args.config, config_type="inference")
    """
    logging related part
    """
    logging_related(output_path=output_path, debug=conf.general.debug, training=False)
    main(conf)
    end = timer()
    logging.info("Total time used: {:.1f}".format(end - start))
