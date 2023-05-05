import torch
from torch import nn


class HierarchicalClustering(nn.Module):
    
    def __init__(self, num_levels, num_centers, sigma, embedding_size):

        super().__init__()

        self.num_levels = num_levels
        self.num_centers = num_centers
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
            for j in range(self.num_centers[i]):
                center_name = "level_" + str(i) + "_center_" + str(j+1)
                self.cluster_centers[center_name] = nn.Parameter(
                    data=torch.rand(self.embedding_size), requires_grad=True)
                

    def assignment_step(self, h_i, level):

        # level starts from 1, not 0, i.e. 1,2,3 for 4 levels
        p_i = torch.zeros(self.num_centers[level-1], self.num_centers[level])

        for i, h_i_cluster in enumerate(h_i):  # test we get a correct h_i_cluster each time
            assignment_scores = torch.zeros(self.num_centers[level])

            for center_idx in range(self.num_centers[level]):
                center_name = "level_" + str(level) + "_center_" + str(center_idx+1)

                assignment_scores[center_idx] = -torch.sum(torch.square(
                    h_i_cluster - self.cluster_centers[center_name]) /
                    (2.0 * self.sigma))

            p_i[i][:] = torch.softmax(assignment_scores, dim=0)  # test it runs ok for all levels

        return p_i


    def update_step(self, p_i, h_i, level):

        # level starts from 1, not 0, i.e. 1,2,3 for 4 levels
        h_i_next = torch.zeros(self.num_centers[level], self.embedding_size)

        for i in range(self.num_centers[level]):
            layer_name = "linear_" + str(level)

            for j in range(self.num_centers[level-1]):
                h_i_next[i] += (p_i[j][i] * torch.tanh(self.linear_layers[layer_name](h_i[j])))

        return h_i_next
 

    def forward(self, task_embedding):
        
        h_level = task_embedding

        for level in range(1, self.num_levels):
            p_level = self.assignment_step(h_level, level)
            h_level = self.update_step(p_level, h_level, level)

        return h_level


def build_cluster_network(num_levels, num_centers, sigma, embedding_size, device):
    cluster_network = HierarchicalClustering(num_levels,
                                             num_centers,
                                             sigma,
                                             embedding_size).to(device)
    return cluster_network