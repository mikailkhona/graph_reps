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
from plot import plot_embeddings_adjacencies
import argparse
import pdb
import sys
from utils import cleanup, set_seed, initialize_and_log
import pdb

from losses import ContrastiveLoss

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('--deploy', action='store_true', help='foo help')
parser.add_argument('--niter', help='foo help')
parser.add_argument('--N', help='foo help')
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
p = float(args.p)
Ne = int(args.Ne)
graph_type = args.graph_type
loss_fn = args.loss_fn

if deploy: 
    print("Deploying with wandb")

# initialize_and_log(deploy, tag='scratch', wandb_project_name='graph_contrastive')

set_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

run_string = "niter_{niter}_N_{N}_p_{p}_Ne_{Ne}_G_{graph_type}"

prefix = "./out/" + run_string

eval_iters = 50
entropy_log_iters = 10
nruns = 3
lambda_entropy_reg = 0.1
train_batch_size = 4096
test_batch_size = 1000
store = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device=='cuda':
    scaler = torch.cuda.amp.GradScaler()

if graph_type == "bernoulli":
    G = graphs.create_bernoulli_graph(num_nodes = N, p = p)
elif graph_type == "spoke":
    G = graphs.create_spoke_graph(num_nodes = N)
elif graph_type == "ring":
    G = graphs.create_ring_graph(num_nodes = N)
else: 
    print("Graph type not recognized")
    sys.exit()

print('Graph created, of type ', graph_type)

num_nodes = G.number_of_nodes()
G, best_loss, assigned_split = graphs.compute_policy_degen(G)
tprobs = np.zeros((num_nodes,num_nodes))
for i in range(num_nodes):
    tprobs[i] = G.nodes[i]['tprobs']

model = model_graph.LinearContrastiveModel(num_nodes=num_nodes, Ne=Ne).to(device)

test,_ = graphs.generate_batch(G, batch_size=test_batch_size, split='test')
test = torch.tensor(test, dtype=torch.float32).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

bias1s = []
bias2s = []
losses = []
entropies = []

if loss_fn == 'simple_contrastive':
    criterion = ContrastiveLoss()

elif loss_fun == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()

for i in range(niter):
    triplets,_ = graphs.generate_batch(G,batch_size=train_batch_size, split='train')
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
                index, bias1, bias2, ps = model_graph.compute_bias(G, model)
                # entropy = -softmaxed_preds*torch.log(softmaxed_preds)
                # entropy = entropy.mean().item()
                # entropies.append(entropy)
                outputs = model(test[:,:2])
                loss_value = criterion(outputs, test[:,2]).item()

            elif loss_fn == 'simple_contrastive':
                bias1, bias2, ps = model_graph.compute_bias_contrastive(G, model)
                outputs = model(test)
                loss = - torch.diag(outputs) + torch.logsumexp(outputs, dim=1)

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

        # ipdb.set_trace()
        # figure1 = plot_embeddings_adjacencies(model.Wu.weight.data.cpu().detach().numpy(), model.We.weight.data.cpu().detach().numpy(), model.V.weight.data.cpu().detach().numpy(), G)

        if deploy:
            wandb.log(evals)
            # wandb.log({"embeddings": figure1})

if store:
    if not os.path.exists(prefix):

        os.makedirs(prefix)

    np.savez(prefix + "/iter%d"%ii, np.array(losses), index , np.array(bias1s), np.array(bias2s), np.array([best_loss]), np.array(tprobs))