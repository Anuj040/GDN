import glob
import os

import kagglehub
import pandas as pd


def download_cmapss():
    cmapss_dir = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    train_path = glob.glob(
        os.path.join(cmapss_dir, "**", "train_FD001.txt"), recursive=True
    )[0]
    cols = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    SENSORS = [
        "s2",
        "s3",
        "s4",
        "s7",
        "s8",
        "s9",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
        "s17",
        "s20",
        "s21",
    ]
    df["maxcyc"] = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = df["maxcyc"] - df["cycle"]
    os.makedirs(os.path.join("data", "cmapss"), exist_ok=True)
    df.to_csv(os.path.join("data", "cmapss", "data.csv"), index=False)
    return df


def get_df() -> pd.DataFrame:
    csv_path = os.path.join("data", "cmapss", "data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(os.path.join("data", "cmapss", "data.csv"))
    else:
        return download_cmapss()


if __name__ == "__main__":
    download_cmapss()
