import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import jax
import time
import optax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from scipy.special import softmax

#Generate start-goal-next triplets
def generate_batch(G,N):
    num_nodes = G.number_of_nodes()
    goals = np.random.choice(num_nodes, N)
    starts =  np.random.choice(num_nodes, N)
    goals= np.zeros(N,dtype=int)
    starts= np.zeros(N,dtype=int)
    nexts=  np.zeros(N,dtype=int)
    for i in range(N):
        goal,start = np.random.choice(num_nodes,2,replace=False)
        goals[i] = goal
        starts[i] = start
        nexts[i] = np.random.choice(num_nodes, p=G.nodes[goals[i]]['policy'][starts[i]])

    triplets_raw = np.concatenate((starts[:,None],goals[:,None],nexts[:,None]),axis=1)
    triplets = np.eye(num_nodes)[triplets_raw]
    return triplets, triplets_raw

@jit
def forward(params, inputs):
    We = params[0]
    V = params[1]
    Wu = params[2]
    vstart = jnp.matmul(inputs[:,0],We)
    vgoal = jnp.matmul(inputs[:,1],We)
    ugoal = jnp.matmul(vgoal,V)
    return jnp.matmul(vstart + ugoal, Wu.T)

def get_best_prediction(params, inputs):
    outputs = forward(params,inputs)
    label_preds = jax.nn.softmax(outputs)
    return jnp.argmax(label_preds,axis=-1)

def get_label_probs(params, inputs, labels):
    outputs = forward(params,inputs)
    label_preds = jax.nn.softmax(outputs)
    label_probs = jnp.sum(label_preds*labels,axis=-1)
    return label_probs

@jit
def loss(params, inputs, labels):
    return -jnp.mean(jnp.log(get_label_probs(params, inputs, labels)))

@jit
def update(params, x, y, lr = 1e-1, w_decay = 1e-6):
    grads = grad(loss,argnums=0)(params, x, y)
    return [(w - lr * dw - w_decay*w) for (w), (dw) in zip(params, grads)]

def compute_bias(G,params):
    num_nodes = G.number_of_nodes()
    #print(G.nodes())
    bias1 = []
    bias2 = []
    index = []
    triplets_raw_all = []
    ps = np.zeros((num_nodes,num_nodes))
    t1 = time.time()

    for start in range(num_nodes):
        for goal in range(num_nodes):
            p = G.nodes[goal]['policy'][start]
            ent = -np.mean(p*np.log(p + 1e-10))
            if ent > 1e-5:
                start_neighbors = list(G.neighbors(start))
                filt1 =  G.nodes[goal]['distance_to_goal'][start_neighbors] == G.nodes[goal]['distance_to_goal'][start] - 1
                filt2 =  G.nodes[goal]['distance_to_goal'][start_neighbors] == G.nodes[goal]['distance_to_goal'][start]
                opt = [start_neighbors[i] for i,f in enumerate(start_neighbors) if filt1[i]]
                subopt = [start_neighbors[i] for i,f in enumerate(start_neighbors) if filt2[i]]

                for ni,next in enumerate(opt):
                    triplets_raw = np.array([start,goal,next])
                    triplets = np.eye(num_nodes)[triplets_raw]
                    index += [[start,goal,next,0]]
                    triplets_raw_all.append(triplets_raw)
                    
                for ni,next in enumerate(subopt):
                    triplets_raw = np.array([start,goal,next])
                    triplets = np.eye(num_nodes)[triplets_raw]

                    index += [[start,goal,next,1]]
                    triplets_raw_all.append(triplets_raw)
    t2 = time.time()

    triplets_raw_all = np.array(triplets_raw_all)
    index = np.array(index)
    triplets = np.eye(num_nodes)[triplets_raw_all]
    outputs = forward(params,triplets[:,:2])
    label_probs = softmax(outputs,axis=1)

    t3 = time.time()
    for start in range(num_nodes):
        for goal in range(num_nodes):
            filt1 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 0)
            filt2 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 1)

            if np.sum(filt1) > 0 and np.sum(filt2) > 0:
                #print(index[filt1,2],index[filt2,2],label_probs[filt1].shape,label_probs[filt2].shape)
                ps_opt = np.mean((label_probs[filt1,index[filt1,2]]))
                ps_subopt = np.mean((label_probs[filt2,index[filt2,2]]))
                #ptot = np.sum(label_probs[filt1,index[filt1,2]]) + np.sum(label_probs[filt2,index[filt2,2]])
                bias1 += [np.log(ps_opt/ps_subopt)]
                bias2 += [ps_opt-ps_subopt]
                #print(ptot)

            ps[start,goal] = np.sum(label_probs[filt1 + filt2,index[filt1 + filt2,2]])


    t4 = time.time()
    #print('Time to compute bias:', t4-t3, t3-t2, t2-t1)

    return index, np.array(bias1), np.array(bias2), ps

