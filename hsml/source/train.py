import torch

import model_builder


def main():
    num_levels = 4
    num_clusters = [4 ,2, 1]
    embedding_size = 128
    device = torch.device("cpu")
    network = model_builder.build_cluster_network(num_levels,
                                                  num_clusters,
                                                  embedding_size,
                                                  device)
    
    for key, value in network.named_parameters():
        print(key, value.shape)


if __name__ == "__main__":
    main()
