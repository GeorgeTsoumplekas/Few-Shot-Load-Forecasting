"""This module contains a function that finds the size of the tasks' embeddings and then
updates the configuration file of the hsml meta-learner. This is useful when we want to
create the complete pipeline from creating the embeddings to getting the final results of
HSML without having to manually intervene between the two.

Typical usage example:
python3 update_config.py --config "path/to/config.yaml" \
                         --train_embeddings_path "path/to/train_embeddings.json"
"""

import argparse
import json
import yaml


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_embeddings_path', dest='train_embeddings_path')
    parser.add_argument('--config', '-C', dest='config_filepath')
    args = parser.parse_args()

    # Embeddings of train tasks
    train_embeddings_path = args.train_embeddings_path
    with open(train_embeddings_path, 'r', encoding='utf8') as stream:
        train_embeddings = json.load(stream)

    # Configuration file
    config_filepath = args.config_filepath
    with open(config_filepath, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    # All tasks (both train and test) have the same embedding size so we can just examine
    # a random embedding.
    embedding_size = len(train_embeddings['000'])

    # Update configuration to contain the correct embedding size
    config['ht_config']['embedding_size'] = embedding_size

    # Save updated configuration
    with open(config_filepath, 'w', encoding='utf8') as stream:
        stream.write(yaml.dump(config, default_flow_style=False))

if __name__ == "__main__":
    main()
