import argparse

from utils import get_folder, simple_download_from_url, download_from_url

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d")
args = parser.parse_args()

DATASET = args.dataset

URL = {
    "compact": "https://www.dcc.fc.up.pt/~ltorgo/Regression",
    "abalone": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression",
    "yearprediction": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/",
    "powerplant": "https://archive.ics.uci.edu/static/public/294",
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "w8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary",
    "higgs": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/",
    "amazon": "https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews",
}
FILES = {
    "compact": ["compact.tar.gz"],
    "abalone": ["abalone"],
    "yearprediction": ["YearPredictionMSD.bz2", "YearPredictionMSD.t.bz2"],
    "powerplant": ["combined+cycle+power+plant.zip"],
    "a9a": ["a9a", "a9a.t"],
    "w8a": ["w8a", "w8a.t"],
    "amazon": [""],
    "higgs": ["HIGGS.csv.gz"],
}

output_folder = get_folder(f"data/raw/{DATASET}")

for file in FILES[DATASET]:
    file_url = f"{URL[DATASET]}/{file}"
    file_path = f"{output_folder}/{file}"

    if file == "":
        raise ValueError("Cannot be downloaded. Check repo for specific instructions.")

    try:
        download_from_url(file_url, file_path)
    except KeyError:
        simple_download_from_url(file_url, file_path)
