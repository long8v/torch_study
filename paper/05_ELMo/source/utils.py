import yaml

def read_yaml(file):
    return yaml.load(open(file, 'r'))