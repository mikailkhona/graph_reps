import graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time
import sys
import os
import wandb
import model_graph
import random
import torch.nn as nn
from plot import plot_embeddings_adjacencies
import argparse
from losses import NTXEntLoss
import pdb


def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)

def open_log(deploy, tag):
    os.makedirs('logs/' + tag, exist_ok=True)
    if deploy:
        fname = 'logs/' + tag + '/' + wandb.run.id + ".log"
        fout = open(fname, "a", 1)
        sys.stdout = fout
        sys.stderr = fout
        return fout

def init_wandb(deploy, wandb_project_name):
    if deploy:
        print('Initializing wandb project')
        wandb.init(project=wandb_project_name)
        wandb.run.name = wandb.run.id
        wandb.run.save()

def cleanup(deploy, fp):
    if deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()



# # Create the parser
# parser = argparse.ArgumentParser(description="Example script.")

# # Add an argument
# parser.add_argument('--deploy', help='foo help')
# parser.add_argument('--niter', help='foo help')
# parser.add_argument('--N', help='foo help')
# parser.add_argument('--p', help='foo help')
# parser.add_argument('--Ne', help='foo help')
# parser.add_argument('--graph_type', help='foo help')

# # Parse the arguments
# args = parser.parse_args()

deploy = False
niter = 100000
N = 100
p = 0.1
Ne = 64
graph_type = "bernoulli"

# Initialize wandb project
init_wandb(deploy, wandb_project_name = "graph_model")
set_seed(0)


# np.set_printoptions(precision = 3, suppress = True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


run_string = "niter_{niter}_N_{N}_p_{p}_Ne_{Ne}_G_{graph_type}"
prefix = "./out/" + run_string

# create log file
fp = open_log(deploy, tag=run_string)

eval_iters = 50
entropy_log_iters = 10
nruns = 3
lambda_entropy_reg = 0.1
store = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    scaler = torch.cuda.amp.GradScaler()

for ii in range(nruns):
    if graph_type == "bernoulli":
        G = graphs.create_bernoulli_graph(num_nodes = N, p = p)
    elif graph_type == "spoke":
        G = graphs.create_spoke_graph(num_nodes = N)
    elif graph_type == "ring":
        G = graphs.create_ring_graph(num_nodes = N)
    else: 
        print("Graph type not recognized")
        break

    num_nodes = G.number_of_nodes()
    G, best_loss = graphs.compute_policy_degen(G)
    tprobs = np.zeros((num_nodes,num_nodes))

    for i in range(num_nodes):
        tprobs[i] = G.nodes[i]['tprobs']
    model = model_graph.LinearContrastiveModel(num_nodes=num_nodes, Ne=Ne).to(device)

    test,_ = graphs.generate_batch(G, batch_size=1000)
    test = torch.tensor(test, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    bias1s = []
    bias2s = []
    losses = []
    entropies = []
    loss_fn = 'simple_contrastive'
    # criterion = nn.CrossEntropyLoss()
    # criterion = NTXEntLoss()

    for i in range(niter):
        triplets,_ = graphs.generate_batch(G,128)
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
                pdb.set_trace()
                preds = model(triplets)
            elif loss_fn == 'cross_entropy':
                preds = model(batch)
           

            # Compute loss
            # cross entropy loss
            if loss_fn =='cross_entropy':
                loss = criterion(preds, labels)

            # simple contrastive loss
            elif loss_fn=='simple_contrastive':
                loss = - torch.diag(preds) + torch.logsumexp(preds, dim=1)
                # softmaxed_preds = F.softmax(preds, dim=-1)

                # # argmax'ed labels
                # label_idxs = torch.argmax(labels,dim=1)

                # #random labels of shape (batch, num_nodes)
                # random_idxs = torch.randint(low=0, high=softmaxed_preds.shape[1], size=(softmaxed_preds.shape[0],))

                # loss = torch.log(softmaxed_preds[torch.arange(softmaxed_preds.shape[0]),label_idxs]) + torch.log(1-(softmaxed_preds[torch.arange(softmaxed_preds.shape[0]), random_idxs])) + lambda_entropy_reg*(softmaxed_preds*torch.log(softmaxed_preds)).mean()
            
            elif loss_fn=='unif_align':
                loss,metrics = align_unif(log_lambda, softmaxed_preds[torch.arange(softmaxed_preds.shape[0]),label_idxs], softmaxed_preds[torch.arange(softmaxed_preds.shape[0]), random_idxs], batch_size)


            # if i%500 ==0: pdb.set_trace()

        scaler.scale(loss.mean()).backward()
        # Unscales gradients and calls or skips optimizer.step
        scaler.step(optimizer)
        # Updates the scale for next iteration
        scaler.update()

        # if i%entropy_log_iters==0:
        #     entropy = -softmaxed_preds*torch.log(softmaxed_preds)
        #     entropy = entropy.mean().item()
        #     entropies.append(entropy)

        if i%eval_iters==0:
            '''
            eval model and log to wandb
            '''
            with torch.no_grad():
                index, bias1, bias2, ps = model_graph.compute_bias(G, model)
                
                if loss_fn == 'cross_entropy':
                    outputs = model(test[:,:2])
                    loss_value = criterion(outputs, test[:,2]).item()
                elif loss_fn == 'simple_contrastive':
                    outputs = model(test)
                    loss = - torch.diag(outputs) + torch.logsumexp(outputs, dim=1)
                # softmaxed_preds = F.softmax(outputs, dim=-1)
                # label_idxs = torch.argmax(test[:,2],dim=1)

                # # simple contrastive loss
                # random_idxs = torch.randint(low=0, high=softmaxed_preds.shape[1],size=(softmaxed_preds.shape[0],))
                # loss = torch.log(softmaxed_preds[torch.arange(softmaxed_preds.shape[0]),label_idxs]) + torch.log(1-(softmaxed_preds[torch.arange(softmaxed_preds.shape[0]), random_idxs]))
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
