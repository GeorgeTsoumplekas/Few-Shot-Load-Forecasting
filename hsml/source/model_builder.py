import torch
from torch import nn


class HierarchicalClustering(nn.Module):
    
    def __init__(self, num_levels, num_clusters, sigma, embedding_size):

        super().__init__()

        self.num_levels = num_levels
        self.num_clusters = num_clusters
        self.embedding_size = embedding_size
        self.sigma = sigma

        self.cluster_centers = nn.ParameterDict()
        self.linear_layers = nn.ModuleDict()

        for i in range(1, self.num_levels):
            layer_name = "linear_" + str(i)
            self.linear_layers[layer_name] = nn.Linear(in_features=self.embedding_size,
                                                       out_features=self.embedding_size,
                                                       bias=True)

        # TODO: determine how to initialize cluster centers
        for i in range(1, self.num_levels):
            for j in range(num_clusters[i-1]):
                center_name = "layer_" + str(i) + "_center_" + str(j+1)
                self.cluster_centers[center_name] = nn.Parameter(
                    data=torch.ones(self.embedding_size), requires_grad=True)
                

    def assignment_step(self, h_i, layer):
        pass


    def update_step(self):
        pass
                
    
    def forward(self, task_embedding):
        pass


def build_cluster_network(num_levels, num_clusters, embedding_size, device):
    cluster_network = HierarchicalClustering(num_levels, num_clusters, embedding_size).to(device)
    return cluster_network