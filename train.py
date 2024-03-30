import graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import jax
import time
import sys
import os
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
from model import *

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ndevices = jax.local_device_count()
print(jax.devices())
# num_devices = len(jax.devices())
device = 0

jax.config.update("jax_default_device", jax.devices()[device])
key = random.PRNGKey(device)
np.random.seed(device)

np.set_printoptions(precision = 3, suppress = True)

niter = int(sys.argv[1])
N  = int(sys.argv[2])
p = float(sys.argv[3])
Ne = int(sys.argv[4])

graph_type = sys.argv[6]

prefix = "./outs/I%08d_N%d_p%.3f_Ne%d_" %(niter,N,p,Ne) + "G" + graph_type

nruns = 1
store= False

for ii in range(nruns):
    run = num_devices*ii + device

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
    
    We = np.random.randn(num_nodes,Ne)/np.sqrt(Ne)
    V = np.random.randn(Ne,Ne)/np.sqrt(Ne)
    Wu = np.random.randn(num_nodes,Ne)/np.sqrt(Ne)

    params = [We,V,Wu]

    test,_ = generate_batch(G,1000)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state, batch, labels):
        loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    bias1s = []
    bias2s = []
    losses = []

    for i in range(niter):
        triplets,_ = generate_batch(G,128)
        params, opt_state, loss_value = step(params, opt_state, triplets[:,:2], triplets[:,2])
        
        if i%100==0:
            index, bias1,bias2,ps = compute_bias(G,params)
            
            loss_value = loss(params,test[:,:2],test[:,2])

            bias1s.append(bias1)
            bias2s.append(bias2)
            losses.append(loss_value)

            print("Niter %05d"%i,"Loss %.2f"%loss_value,"Bias1  %.3f"%np.mean(bias1),"Bias2  %.3f"%np.mean(bias2), \
                  r"$\Delta$Bias1  %.3f"%np.std(bias1), "BestLoss %.3f"%best_loss)

    if store:
        if not os.path.exists(prefix):

            os.makedirs(prefix)

        np.savez(prefix + "/iter%d"%run, np.array(losses), index , np.array(bias1s), np.array(bias2s), np.array([best_loss]), np.array(tprobs))