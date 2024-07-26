import configparser

def make_default_config():
    config = configparser.ConfigParser()

    config['embed_dim'] = 1024
    config['model'] = {
        'name': 'PointBERT',
        'in_channel': 6,
        'out_channel': config['embed_dim'],
        'embedding_channel': 1024,
        'voxel_size': 0.02
    }

    with open('configs/config.ini', 'w') as configfile:
        config.write(configfile)


def load_config():
    config = configparser.ConfigParser()
    config.read('configs/config.ini')
    return config


def dump_config(path, config):
    config = configparser.ConfigParser()
    with open(path, 'w') as fp:
        config.write(config=config, f=fp)