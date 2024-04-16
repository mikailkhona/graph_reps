import graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time
import sys
import os
import model_graph
import random
import torch.nn as nn
from plot_graph import plot_embeddings_adjacencies, plot_neighbour, plot_and_fit_short_bias
import argparse
from utils import cleanup, set_seed, initialize_and_log
import pdb
import wandb
from losses import ContrastiveLoss

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('--deploy', action='store_true', help='foo help')
parser.add_argument('--niter', help='foo help')
parser.add_argument('--N', help='foo help')
parser.add_argument('--frac', help='foo help')
parser.add_argument('--p', help='foo help')
parser.add_argument('--Ne', help='foo help')

parser.add_argument('--graph_type', help='foo help')
parser.add_argument('--loss_fn', help='foo help')

# Parse the arguments
args = parser.parse_args()

#collect arguments in variables
deploy = args.deploy
niter = int(args.niter)
N = int(args.N)
frac = float(args.frac)
p = float(args.p)
Ne = int(args.Ne)
graph_type = args.graph_type
loss_fn = args.loss_fn


if deploy: 
    print("Deploying with wandb")

initialize_and_log(deploy, tag='scratch', wandb_project_name='graph_contrastive')

set_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

run_string = f"niter_{niter}_N_{N}_p_{p}_Ne_{Ne}_G_{graph_type}"

prefix = "./out/" + run_string

eval_iters = 50
nruns = 3
lambda_entropy_reg = 0.1
train_batch_size = 256
test_batch_size = 1000
store = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device=='cuda':
    scaler = torch.cuda.amp.GradScaler()

if graph_type == "bernoulli":
    G = model_graph.create_bernoulli_graph(num_nodes = N, p = p)
elif graph_type == "spoke":
    G = model_graph.create_spoke_graph(num_nodes = N)
elif graph_type == "ring":
    G = model_graph.create_ring_graph(num_nodes = N)
else: 
    print("Graph type not recognized")
    sys.exit()

print('Graph created, of type ', graph_type)

num_nodes = G.number_of_nodes()
G, best_loss, assigned_split = graphs.compute_policy_degen_modified(G, frac=frac)
tprobs = np.zeros((num_nodes,num_nodes))

for i in range(num_nodes):
    tprobs[i] = G.nodes[i]['tprobs']
# generate eval data
test,_ = graphs.generate_batch(
                            G, 
                            assigned_split=assigned_split,
                            batch_size=test_batch_size, 
                            split='test'
                            )

test = torch.tensor(test, dtype=torch.float32).to(device)


bias1s = []
bias2s = []
losses = []
entropies = []

if loss_fn == 'simple_contrastive':
    criterion = ContrastiveLoss()
    model = model_graph.LinearContrastiveModel(num_nodes=num_nodes, Ne=Ne).to(device)


elif loss_fn == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
    model = model_graph.LinearModel(num_nodes=num_nodes, Ne=Ne).to(device)
    
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(niter):

    triplets,_ = graphs.generate_batch(
                                    G, 
                                    assigned_split=assigned_split,
                                    batch_size=train_batch_size, 
                                    split='train'
                                    )
                                    
    # triplets: (batch, (goal, start, next), num_nodes)
    triplets = torch.tensor(triplets, dtype=torch.float32).to(device)
    # create input and labels
    batch = triplets[:,:2,:]
    # last index is next
    labels = triplets[:,2,:]
    # zero out gradients
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        # Forward pass
            # preds = shape (batch, num_nodes)
        if loss_fn == 'simple_contrastive':
            preds = model(triplets)
            loss = criterion(preds)

        elif loss_fn == 'cross_entropy':
            preds = model(batch)
            loss = criterion(preds, labels)

    scaler.scale(loss.mean()).backward()
    # Unscales gradients and calls or skips optimizer.step
    scaler.step(optimizer)
    # Updates the scale for next iteration
    scaler.update()


    if i%eval_iters==0:
        '''
        eval model and optionally log to wandb
        '''
        print(loss.mean().item())

        with torch.no_grad():
            
            
            if loss_fn == 'cross_entropy':
                index, bias1, bias2, ps = graphs.compute_bias(G, model)
                # entropy = -softmaxed_preds*torch.log(softmaxed_preds)
                # entropy = entropy.mean().item()
                # entropies.append(entropy)
                outputs = model(test[:,:2])
                loss = criterion(outputs, test[:,2])

            elif loss_fn == 'simple_contrastive':
                bias1, bias2, ps = graphs.compute_bias_contrastive(G, model)
                outputs = model(test)
                # loss = - torch.diag(outputs) + torch.logsumexp(outputs, dim=1)
                loss = criterion(outputs)

            loss_value = loss.mean().item()


        bias1s.append(bias1)
        bias2s.append(bias2)
        losses.append(loss_value)

        evals = {
                "loss":loss_value,
                "bias1":np.mean(bias1), 
                "bias2":np.mean(bias2), 
                "best_loss":best_loss,
                }

        non_bias_weights = {} 
        for name, param in model.named_parameters():
            if "bias" not in name:
                non_bias_weights[name] = param.data.detach().cpu().numpy().copy()

        figure1, start_next, goal_next = plot_embeddings_adjacencies(non_bias_weights, G, loss_fn)
        figure2, neighbors_mean, nneighbors_mean = plot_neighbour(G, start_next)
        figure3, ps, r2 = plot_and_fit_short_bias(G, goal_next)

        if deploy:
            wandb.log(evals)
            wandb.log({"embeddings_and_adjacency": figure1})
            wandb.log({"neighbour": wandb.Image(figure2)})
            wandb.log({"short_path_bias": wandb.Image(figure3)})

if store:
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    np.savez(prefix + "/iter%d"%ii, np.array(losses), index , np.array(bias1s), np.array(bias2s), np.array([best_loss]), np.array(tprobs))