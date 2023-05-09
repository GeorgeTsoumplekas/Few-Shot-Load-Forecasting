import argparse
import json
import os
import yaml

import torch

import engine
import model_builder
import utils


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

    # train_embeddings_filename = "../../data/mini_iONA_train_embeddings.json"
    # with open(train_embeddings_filename, 'r', encoding='utf8') as stream:
    #     embeddings = json.load(stream)

    # task_embedding = torch.tensor(embeddings['002']).unsqueeze(dim=0)

    # num_levels = 4
    # num_centers = [1, 4 ,2, 1]
    # embedding_size = 134
    # device = torch.device("cpu")
    # sigma = 2.0

    # network = model_builder.build_cluster_network(num_levels,
    #                                               num_centers,
    #                                               sigma,
    #                                               embedding_size,
    #                                               device)

    # h_i_final = network(task_embedding)

    # for key, value in network.named_parameters():
    #     print(key, value.shape)

    args = {
        'train_epochs': 5,
        'task_batch_size': 1,
        'sample_batch_size': 1,
        'lstm_hidden_units': 16,
        'init_learning_rate': 1e-3,
        'meta_learning_rate': 1e-4,
        'eta_min': 1e-6,
        'num_inner_steps': 3,
        'second_order': False,
        'second_to_first_order_epoch': 5,
        'num_levels': 4,
        'num_centers': [1, 4 ,2, 1],
        'sigma': 2.0,
        'embedding_size': 134
    }

    data_config = {
        'day_measurements': 96,
        'week_num': 1,
        'pred_days': 1,
        'test_days': 7
    }

    meta_learner = engine.build_meta_learner(args=args,
                                                 data_config=data_config)


if __name__ == "__main__":
    main()
