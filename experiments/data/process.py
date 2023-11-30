import argparse
import gzip
import zipfile
import tarfile
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm

from utils import get_folder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="Dataset name.")
parser.add_argument(
    "--n-samples",
    "-n",
    default=None,
    type=int,
    help="Number of samples for big datasets.",
)
parser.add_argument(
    "--seed", "-s", default=0, help="Seed for resampling dataset. Default is 0."
)
args = parser.parse_args()


def process(dataset, n_samples=None, seed=0):
    if dataset == "compact":
        with tarfile.open(f"data/raw/{dataset}/compact.tar.gz") as tar:
            full = np.loadtxt(
                tar.extractfile("ComputerActivity/cpu_act.data"), delimiter=","
            )

        X, y = full[:, 0:21], full[:, 21]
        data = pd.DataFrame(X)
        data["target"] = pd.DataFrame(y)

    if dataset == "abalone":
        X, y = load_svmlight_file(f"data/raw/{dataset}/{dataset}")
        data = pd.DataFrame(X.toarray())
        data["target"] = pd.DataFrame(y)

    if dataset == "powerplant":
        archive = zipfile.ZipFile(
            f"data/raw/{dataset}/combined+cycle+power+plant.zip", "r"
        )
        with archive.open("CCPP/Folds5x2_pp.xlsx") as file:
            df = pd.read_excel(file, sheet_name=None, index_col=None, header=0)

        df = pd.concat(df.values())
        X, y = df.iloc[:, 0:4], df.iloc[:, 4]
        data = X
        data.columns = range(data.columns.size)
        data["target"] = pd.DataFrame(y)

    if dataset == "proteinstructure":
        df = pd.read_csv(
            "data/raw/proteinstructure/physicochemical+properties+of+protein+tertiary+structure.zip/CASP.csv",
            header=None,
        )

        X, y = df.iloc[:, 0:4], df.iloc[:, 4]
        data = X
        data.columns = range(data.columns.size)
        data["target"] = pd.DataFrame(y)

    if dataset == "a9a":
        X, y = load_svmlight_file(f"data/raw/{dataset}/{dataset}", n_features=123)
        X_t, y_t = load_svmlight_file(f"data/raw/{dataset}/{dataset}.t", n_features=123)

        X = X.toarray()
        X_t = X_t.toarray()

        X = np.vstack((X, X_t))
        y = np.hstack((y, y_t))

        data = pd.DataFrame(X)
        data["target"] = pd.DataFrame(y)

    if dataset == "w8a":
        X, y = load_svmlight_file(f"data/raw/{dataset}/{dataset}", n_features=300)
        X_t, y_t = load_svmlight_file(f"data/raw/{dataset}/{dataset}.t", n_features=300)

        X = X.toarray()
        X_t = X_t.toarray()

        X = np.vstack((X, X_t))
        y = np.hstack((y, y_t))

        data = pd.DataFrame(X)
        data["target"] = pd.DataFrame(y)

    if dataset == "yearprediction":
        X, y = load_svmlight_file(
            f"data/raw/{dataset}/YearPredictionMSD.bz2", n_features=90
        )
        X_t, y_t = load_svmlight_file(
            f"data/raw/{dataset}/YearPredictionMSD.t.bz2", n_features=90
        )

        X = X.toarray()
        X_t = X_t.toarray()

        X = np.vstack((X, X_t))
        y = np.hstack((y, y_t))

        data = pd.DataFrame(X)
        data["target"] = pd.DataFrame(y)

    if dataset == "ailerons":
        with tarfile.open(f"data/raw/{dataset}/ailerons.tgz") as tar:
            train = np.loadtxt(tar.extractfile("Ailerons/ailerons.data"), delimiter=",")
            test = np.loadtxt(tar.extractfile("Ailerons/ailerons.test"), delimiter=",")

        stacked = np.vstack((train, test))
        X, y = stacked[:, 0:40], stacked[:, 40].reshape(-1, 1)

        data = pd.DataFrame(X)
        data["target"] = y

    if dataset == "higgs":
        with gzip.open(f"data/raw/{dataset}/HIGGS.csv.gz") as gz:
            df = pd.read_csv(gz, header=None)

        if n_samples:
            df = df.sample(n=n_samples, random_state=seed)

        X, y = df.iloc[:, 1:], df.iloc[:, 0]

        data = pd.DataFrame(X.to_numpy())
        data["target"] = y.to_numpy().reshape(-1, 1)

    if dataset == "amazon":
        nltk.download("stopwords")

        data = pd.read_csv(f"data/raw/{dataset}/Reviews.csv")

        # https://www.kaggle.com/code/shashanksai/text-preprocessing-using-python.
        # Remove "neutral" category.
        data_score_removed = data[data["Score"] != 3]

        # Define new score vector.
        score_update = data_score_removed["Score"]

        # Map scores to sentiment (0 or 1).
        sentiment = score_update.map(lambda x: 0 if x < 3 else 1)

        # Insert sentiment on score column.
        pd.options.mode.chained_assignment = None
        data_score_removed.loc[:, ["Score"]] = sentiment

        # Drop duplicates where needed.
        data_drop_duplicates = data_score_removed.drop_duplicates(
            subset={"UserId", "ProfileName", "Time", "Text"}
        )

        # Validate entries based on definition of numerator and denominator
        # features.
        final = data_drop_duplicates[
            data_drop_duplicates["HelpfulnessNumerator"]
            <= data_drop_duplicates["HelpfulnessDenominator"]
        ]

        if n_samples:
            final = final.sample(n=n_samples, random_state=seed)

        X_text = final["Text"]
        y = final["Score"].to_numpy()

        sent = []
        snow = SnowballStemmer("english")
        for i in tqdm(range(len(X_text))):
            sentence = X_text.iloc[i]
            sentence = sentence.lower()  # Converting to lowercase
            cleaner = re.compile("<.*?>")  # Define the pattern for HTML tags.
            sentence = re.sub(cleaner, " ", sentence)  # Removing HTML tags
            sentence = re.sub(r'[?|!|\'|"|#]', r"", sentence)  # Removing marks.
            sentence = re.sub(r"[.|,|)|(|\|/]", r" ", sentence)  # Removing punctuations

            sequ = " ".join(
                [
                    snow.stem(word)
                    for word in sentence.split()
                    if word not in stopwords.words("english")
                ]
            )
            sent.append(sequ)

        X_text = np.array(sent)
        data = pd.DataFrame(X_text)
        data["target"] = y

    return data


DATASET = args.dataset
N_SAMPLES = args.n_samples
SEED = args.seed

data = process(DATASET, N_SAMPLES, SEED)

output_folder = get_folder(f"data/processed")
data.to_csv(f"{output_folder}/{DATASET}.csv", index=False)
print(f"-saved: {output_folder}/{DATASET}.csv")
