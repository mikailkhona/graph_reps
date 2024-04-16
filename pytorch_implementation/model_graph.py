
import numpy as np
import torch 
import torch.nn.functional as F
import torch.nn as nn
import pdb
import networkx as nx
import random

def create_linear_graph(num_nodes = 5):
    G = nx.Graph()
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
    return G

def create_ring_graph(num_nodes = 5):
    G = nx.Graph()
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
    G.add_edge(num_nodes - 1, 0)
    return G

def create_spoke_graph(num_nodes = 5):
    G = nx.Graph()
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
    G.add_edge(num_nodes - 1, 0)
    for i in range(num_nodes, 2*num_nodes):
        G.add_edge(i - num_nodes, i)
    return G

def create_bernoulli_digraph(num_nodes = 5, p = 0.2):
    A = np.triu(np.random.choice(2, size = (num_nodes, num_nodes), p = [1 - p, p]), k = 1)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G)
    return G

def create_bernoulli_graph(num_nodes = 5, p = 0.2):
    A = np.triu(np.random.choice(2, size = (num_nodes, num_nodes), p = [1 - p, p]), k = 1)
    G = nx.from_numpy_array(A, create_using=nx.Graph)
    G.remove_nodes_from(list(nx.isolates(G)))
    G = nx.convert_node_labels_to_integers(G)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G)
    return G


class LinearModel(nn.Module):
    def __init__(self, num_nodes, Ne):
        super(LinearModel, self).__init__()
        # Initialize linear model components
        self.We = nn.Linear(num_nodes, Ne, bias=False)
        self.Wu = nn.Linear(Ne, num_nodes, bias=False)
        self.V = nn.Linear(Ne, Ne, bias=False)

    def forward(self, inputs):
        vstart = self.We(inputs[:,0])
        vgoal = self.We(inputs[:,1])
        ugoal = self.V(vgoal)
        return self.Wu(vstart + ugoal)
    
class LinearIndependentModel(nn.Module):
    '''
    Different weights used for goal and start
    '''
    def __init__(self, num_nodes, Ne):
        super(LinearIndependentModel, self).__init__()
        # Initialize linear model components
        self.Ws = nn.Linear(num_nodes, Ne, bias=False)
        self.Wu = nn.Linear(Ne, num_nodes, bias=False)
        self.Wg = nn.Linear(num_nodes, Ne, bias=False)

    def forward(self, inputs):
        vstart = self.Ws(inputs[:,0])
        vgoal = self.Wg(inputs[:,1])
        return self.Wu(vstart + vgoal)

class LinearContrastiveModel(nn.Module):

    def __init__(self, num_nodes, Ne):
        super().__init__()
        self.Ws = nn.Linear(num_nodes,Ne)
        self.Wg = nn.Linear(num_nodes,Ne)
        self.Wnext = nn.Linear(num_nodes,Ne)

    def forward(self, inputs):
        '''
        inputs: (batch, 3, num_nodes)
        returns: (batch, batch)
        '''
        vstart = self.Ws(inputs[:,0,:])
        vgoal = self.Wg(inputs[:,1,:])
        vnext = self.Wnext(inputs[:,2,:])
        vstartnext = vstart + vnext
        return torch.einsum('bj,cj->bc',vstartnext,vgoal)

class LinearReducedModel(nn.Module):
    '''
    Same Weights used for goal and start
    '''
    def __init__(self, num_nodes, Ne):
        super(LinearReducedModel, self).__init__()
        # Initialize linear model components
        self.We = nn.Linear(num_nodes, Ne, bias=False)
        self.Wu = nn.Linear(Ne, num_nodes, bias=False)

    def forward(self, inputs):
        vstart = self.We(inputs[:,0])
        vgoal = self.We(inputs[:,1])
        return self.Wu(vstart + vgoal)
    
class MLP(nn.Module):
    def __init__(self, num_nodes, Ne):
        super(MLP, self).__init__()
        # Initialize MLP model components
        self.mlp_start = nn.Sequential(
            nn.Linear(num_nodes, Ne),
            nn.ReLU(),
            nn.Linear(Ne, Ne),
            nn.ReLU(),
            nn.Linear(Ne, Ne)
        )
        self.mlp_goal = nn.Sequential(
            nn.Linear(num_nodes, Ne),
            nn.ReLU(),
            nn.Linear(Ne, Ne),
            nn.ReLU(),
            nn.Linear(Ne, Ne)
        )
        self.Wu = nn.Linear(Ne, num_nodes, bias=False)

    def forward(self, inputs):
        vstart = self.mlp_start(inputs[:,0])
        vgoal = self.mlp_goal(inputs[:,1])
        return self.Wu(vstart + vgoal)


def get_best_prediction(model, inputs):
    '''
    Take in model and inputs, return best prediction using
    pytorch's argmax and softmax
    '''
    outputs = model(inputs)
    label_preds = F.softmax(outputs)
    return torch.argmax(label_preds, dim=-1)

def get_label_probs(model, inputs, labels):
    '''
    Take in model and inputs, return label probabilities
    '''
    outputs = model(inputs)
    label_preds = F.softmax(outputs)
    label_probs = torch.sum(label_preds*labels, dim=-1)
    return label_probs

def calc_entropy(probability_input_tensor):
    '''
    shape: (batch, categorical)
    '''
    log_p = torch.log(probability_input_tensor)
    p_log_p = log_p*probability_input_tensor
    entropy = -p_log_p.mean()
    return entropy