import pandas as pd
import wandb

from datasets import SuperResolutionDataset
from trainer import Trainer
from config import training_args, model_args


def main():
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
