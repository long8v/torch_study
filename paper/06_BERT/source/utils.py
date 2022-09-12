import os
import yaml
import datetime
import pickle


def read_yaml(file):
    return yaml.safe_load(open(file, "r", encoding="utf8"))


def save_yaml(obj, file):
    with open(file, "w") as f:
        return yaml.dump(obj, f)


def mkdir(path):
    os.mkdir(path, exist_ok=True)


def get_now():
    return datetime.datetime.now().strftime("%d%m%H%M")


def read_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)
