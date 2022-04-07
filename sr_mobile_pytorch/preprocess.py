import pandas as pd
import cv2
import random
import os
import json
from glob import glob
from tqdm.auto import tqdm


def train_test_split(df, test_size=0.1):
    valid = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lr, hr = cv2.imread(row["lr"]), cv2.imread(row["hr"])
        if lr.shape[0] * 4 == hr.shape[0] and lr.shape[1] * 4 == hr.shape[1]:
            valid.append(row)

    if len(valid) <= (test_size * len(df)):
        test_df = pd.DataFrame(valid)
    else:
        valid = random.sample(valid, int(test_size * len(df)))
        test_df = pd.DataFrame(valid)

    common = df.merge(test_df, on=["lr", "hr"])

    # source: https://stackoverflow.com/questions/28901683/pandas-get-rows-which-are-not-in-other-dataframe
    train_df = df[(~df.lr.isin(common.lr)) & (~df.hr.isin(common.hr))]

    return train_df, test_df


def main():
    with open("sr_mobile_pytorch/config.json", "r") as f:
        config = json.load(f)
    training_args = config["training_args"]

    random.seed(training_args["seed"])

    train_LR = sorted(glob(f"{training_args['data_lr']}/*"))
    train_HR = sorted(glob(f"{training_args['data_hr']}/*"))

    df = pd.DataFrame(data={"lr": train_LR, "hr": train_HR})
    train_df, test_df = train_test_split(df, test_size=training_args["test_size"])

    outdir = training_args["outdir"]
    datadir = f"{outdir}/data"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    train_df.to_csv(f"{datadir}/train.csv", index=False)
    test_df.to_csv(f"{datadir}/test.csv", index=False)


if __name__ == "__main__":
    main()
