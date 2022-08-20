from source.trainer import *
from source.utils import *

if __name__ == "__main__":
    config = read_yaml("~/torch_study/paper/04_transformer/config.yaml")
    multi30k_trainer(config)
