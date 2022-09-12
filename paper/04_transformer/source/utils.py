import os
import yaml
import datetime


def read_yaml(file):
    return yaml.safe_load(open(file, "r", encoding="utf8"))


def save_yaml(obj, file):
    with open(file, "w") as f:
        return yaml.dump(obj, f)


def mkdir(path):
    os.mkdir(path)


def get_now():
    return datetime.datetime.now().strftime("%d%m%H%M")
