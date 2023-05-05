import argparse
import json
import os
import yaml

import torch

import model_builder


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_dir', dest='train_dir')
    # parser.add_argument('--test_dir', dest='test_dir')
    # parser.add_argument('--config', '-C', dest='config_filepath')
    # args = parser.parse_args()

    # train_dir = args.train_dir
    # test_dir = args.test_dir
    # config_filepath = args.config_filepath
    # with open(config_filepath, 'r', encoding='utf8') as stream:
    #     config = yaml.safe_load(stream)

    train_embeddings_filename = "../../data/mini_iONA_train_embeddings.json"
    with open(train_embeddings_filename, 'r', encoding='utf8') as stream:
        embeddings = json.load(stream)

    task_embedding = torch.tensor(embeddings['002']).unsqueeze(dim=0)

    num_levels = 4
    num_centers = [1, 4 ,2, 1]
    embedding_size = 134
    device = torch.device("cpu")
    sigma = 2.0

    network = model_builder.build_cluster_network(num_levels,
                                                  num_centers,
                                                  sigma,
                                                  embedding_size,
                                                  device)

    h_i_final = network(task_embedding)

    # for key, value in network.named_parameters():
    #     print(key, value.shape)

    


if __name__ == "__main__":
    main()
