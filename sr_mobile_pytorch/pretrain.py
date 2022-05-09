import pandas as pd
import wandb

from sr_mobile_pytorch.datasets import SuperResolutionDataset
from sr_mobile_pytorch.trainer import Trainer
from sr_mobile_pytorch.trainer.utils import load_config


def main():
    config = "sr_mobile_pytorch/config/pretraining_config_x2.json"
    model_args, training_args = load_config(config)

    wandb.init(project=training_args["project"], entity=training_args["entity"])
    wandb.config = {
        "learning_rate": training_args["learning_rate"],
        "epochs": training_args["epochs"],
        "batch_size": training_args["train_batch_size"],
    }

    datadir = f"{training_args['outdir']}/data"
    train_df = pd.read_csv(f"{datadir}/train.csv")
    test_df = pd.read_csv(f"{datadir}/test.csv")

    train_dataset = SuperResolutionDataset(
        train_df, model_args["scale"], training_args["patch_size"], train=True
    )
    test_dataset = SuperResolutionDataset(test_df, train=False)

    trainer = Trainer(model_args, training_args, train_dataset, test_dataset)
    trainer.fit()


if __name__ == "__main__":
    main()
