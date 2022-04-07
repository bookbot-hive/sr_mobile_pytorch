import pandas as pd
import wandb
import json

from sr_mobile_pytorch.datasets import SuperResolutionDataset
from sr_mobile_pytorch.gan_trainer import GANTrainer
from sr_mobile_pytorch.utils import load_config


def main():
    config = "sr_mobile_pytorch/config/finetuning_config.json"
    model_args, training_args = load_config(config)

    wandb.init(project=training_args["project"], entity=training_args["entity"])
    wandb.config = {
        "generator_learning_rate": training_args["generator_learning_rate"],
        "discriminator_learning_rate": training_args["discriminator_learning_rate"],
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

    trainer = GANTrainer(model_args, training_args, train_dataset, test_dataset)
    trainer.fit()


if __name__ == "__main__":
    main()
