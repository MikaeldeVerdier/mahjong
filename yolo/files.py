import yaml

def load_yaml(path):
    with open(path, "r") as file:
        data_loaded = yaml.safe_load(file)

    return data_loaded
