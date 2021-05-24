import yaml

def read_yaml(file):
    return yaml.safe_load(open(file, 'r'))