import yaml
import json

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_json_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config
