from argparse import ArgumentParser
import yaml

import trainer.CNN as CNN

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', default='./config/CNN/default.yaml', type=str, help='Configuration file path')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Configuration file not found. Please check the path to the configuration file.')

    CNN.train(config)