import json
import os
from types import SimpleNamespace

DEFAULT_CONFIG = {
    'embed_dim': 1024,
    'num_points': 10000,
    'model': {
        'name': 'PointBERT',
        'in_channel': 6,
        'out_channel': 1280,
        'embedding_channel': 1280,
        'voxel_size': 0.02
    },
    'data_dir': './data',
    "rendering_width": 800,
    "rendering_height": 800
}



def make_default_config():
    config_path = './utils/configs/config.json'
    
    if not os.path.exists(config_path):
        with open(config_path, 'w') as fp:
            json.dump(DEFAULT_CONFIG, fp, indent=4)
    else:
        with open(config_path, 'r') as fp:
            config = json.load(fp)
        
        updated = False
        for section in DEFAULT_CONFIG:
            if section not in config:
                config[section] = DEFAULT_CONFIG[section]
                updated = True
            elif isinstance(DEFAULT_CONFIG[section], dict):
                for key in DEFAULT_CONFIG[section]:
                    if key not in config[section]:
                        config[section][key] = DEFAULT_CONFIG[section][key]
                        updated = True

        if updated:
            with open(config_path, 'w') as fp:
                json.dump(config, fp, indent=4)

def load_config():
    config_path = './utils/configs/config.json'
    with open(config_path, 'r') as fp:
        config = json.load(fp, object_hook=lambda d: SimpleNamespace(**d))
    return config

def dump_config(path, config):
    with open(path, 'w') as fp:
        json.dump(config, fp, indent=4)

