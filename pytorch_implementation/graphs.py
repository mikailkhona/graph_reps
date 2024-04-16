import networkx as nx
import numpy as np
from itertools import combinations, permutations
import random
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_bias_split(G, model, assigned_split, split):
    ''''
    Compute the bias of the model's 1 step prediction
    '''

    num_nodes = G.number_of_nodes()
    bias1 = []
    bias2 = []
    index = []
    triplets_raw_all = []
    ps = np.zeros((num_nodes,num_nodes))

    # for start in range(num_nodes):
    #     for goal in range(num_nodes):
    for start, goal in assigned_split[split]:
            p = G.nodes[goal]['policy'][start]
            ent = -np.mean(p*np.log(p + 1e-10))
            if ent > 1e-5:
                start_neighbors = list(G.neighbors(start))
                filt1 =  G.nodes[goal]['distance_to_goal'][start_neighbors] == G.nodes[goal]['distance_to_goal'][start] - 1
                filt2 =  G.nodes[goal]['distance_to_goal'][start_neighbors] >= G.nodes[goal]['distance_to_goal'][start]
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

    triplets_raw_all = np.array(triplets_raw_all)
    index = np.array(index)
    triplets = np.eye(num_nodes)[triplets_raw_all]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    triplets = torch.tensor(triplets, dtype=torch.float32).to(device)
    
    outputs = model(triplets[:,:2])
    label_probs = F.softmax(outputs,dim=-1)
    label_probs = label_probs.cpu().detach().numpy()

    for start, goal in assigned_split[split]:
    # for start in range(num_nodes):
    #     for goal in range(num_nodes):
            filt1 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 0)
            filt2 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 1)

            if np.sum(filt1) > 0 and np.sum(filt2) > 0:
        
                ps_opt = np.mean((label_probs[filt1,index[filt1,2]]))
                ps_subopt = np.mean((label_probs[filt2,index[filt2,2]]))

                bias1 += [np.log(ps_opt/ps_subopt)]
                bias2 += [ps_opt-ps_subopt]


            ps[start,goal] = np.sum(label_probs[filt1 + filt2, index[filt1 + filt2,2]])

    return index, np.array(bias1), np.array(bias2), ps


def compute_bias_contrastive(G, model):
    ''''
    Compute the bias of the model's 1 step prediction
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_nodes = G.number_of_nodes()
    bias1 = []
    bias2 = []
    entropies = []

    ps = np.zeros((num_nodes,num_nodes))

    for start in range(num_nodes):
        for goal in range(num_nodes):
            index = []
            triplets_raw_all = []
            p = G.nodes[goal]['policy'][start]
            ent = -np.mean(p*np.log(p + 1e-10))

            if ent > 1e-5:
                start_neighbors = list(G.neighbors(start))
                # take a step towards goal 
                filt1 =  G.nodes[goal]['distance_to_goal'][start_neighbors] == G.nodes[goal]['distance_to_goal'][start] - 1
                # take a step not towards goal
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

                # all triplets have the same start and goal
                triplets_raw_all = np.array(triplets_raw_all)
                index = np.array(index)
                triplets = np.eye(num_nodes)[triplets_raw_all]
                triplets = torch.tensor(triplets, dtype=torch.float32).to(device)

                # get model output for the whole batch, which by construction has same start, goal and all possible next's
                # [start, goal, next]
                outputs = model(triplets)
                # extra the diagonal: the model prediction for each next
                model_preds = torch.diag(outputs)
                # since start, goal is the same and exp of output is \propto the Q value, we can take softmax to get the model's policy function
                label_probs = F.softmax(model_preds,dim=-1)
                label_probs = label_probs.cpu().detach().numpy()

                # label probs is now 1 dimensional, representing the policy for a single start, goal pair
                # same analysis as before for calculating the bias

                # filt is basically index
                filt1 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 0)
                filt2 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 1)

                if np.sum(filt1) > 0 and np.sum(filt2) > 0:
            
                    ps_opt = np.mean((label_probs[filt1]))
                    ps_subopt = np.mean((label_probs[filt2]))

                    bias1 += [np.log(ps_opt/ps_subopt)]
                    bias2 += [ps_opt-ps_subopt]
                    entropy = -np.mean(label_probs*np.log(label_probs))


                    ps[start,goal] = np.sum(label_probs[filt1 + filt2])
                    entropies.append(entropy)

            
    return np.array(bias1), np.array(bias2), ps

def compute_bias(G, model):
    ''''
    Compute the bias of the model's 1 step prediction
    '''

    num_nodes = G.number_of_nodes()
    bias1 = []
    bias2 = []
    index = []
    triplets_raw_all = []
    ps = np.zeros((num_nodes,num_nodes))

    for start in range(num_nodes):
        for goal in range(num_nodes):
            p = G.nodes[goal]['policy'][start]
            ent = -np.mean(p*np.log(p + 1e-10))
            if ent > 1e-5:
                start_neighbors = list(G.neighbors(start))
                filt1 =  G.nodes[goal]['distance_to_goal'][start_neighbors] == G.nodes[goal]['distance_to_goal'][start] - 1
                filt2 =  G.nodes[goal]['distance_to_goal'][start_neighbors] >= G.nodes[goal]['distance_to_goal'][start]
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

    triplets_raw_all = np.array(triplets_raw_all)
    index = np.array(index)
    triplets = np.eye(num_nodes)[triplets_raw_all]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    triplets = torch.tensor(triplets, dtype=torch.float32).to(device)
    
    outputs = model(triplets[:,:2])
    label_probs = F.softmax(outputs,dim=-1)
    label_probs = label_probs.cpu().detach().numpy()


    for start in range(num_nodes):
        for goal in range(num_nodes):
            filt1 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 0)
            filt2 = (index[:,0] == start)*(index[:,1] == goal)*(index[:,3] == 1)

            if np.sum(filt1) > 0 and np.sum(filt2) > 0:
        
                ps_opt = np.mean((label_probs[filt1,index[filt1,2]]))
                ps_subopt = np.mean((label_probs[filt2,index[filt2,2]]))

                bias1 += [np.log(ps_opt/ps_subopt)]
                bias2 += [ps_opt-ps_subopt]


            ps[start,goal] = np.sum(label_probs[filt1 + filt2, index[filt1 + filt2,2]])

    return index, np.array(bias1), np.array(bias2), ps

def compute_distance_to_goals(G):
    '''
    Computes shortest path length to goal for each node to each node in the graph (since it is connect this is always true)
    '''
    for goal in G.nodes():
        G.nodes[goal]['distance_to_goal'] = np.zeros(G.number_of_nodes())
        for node in G.nodes():
            G.nodes[goal]['distance_to_goal'][node] = nx.shortest_path_length(G, node, goal)
    return G



def compute_policy_degen(G, frac=0.8):
    num_nodes = G.number_of_nodes()
    G = compute_distance_to_goals(G)
    for goal in G.nodes():
        G.nodes[goal]['policy'] = np.zeros((num_nodes, num_nodes))
        for j in G.nodes():
            probs = np.zeros(num_nodes)
            if j != goal:
                j_neighbors = list(G.neighbors(j))
                for jn in j_neighbors:
                    if G.nodes[goal]['distance_to_goal'][jn]  == G.nodes[goal]['distance_to_goal'][j] \
                        or G.nodes[goal]['distance_to_goal'][jn]  == G.nodes[goal]['distance_to_goal'][j] - 1:
                        probs[jn] = 1
                if np.sum(probs) == 0:
                    probs += 1
                probs /= np.sum(probs)
            else:
                probs[goal] = 1
                probs /= np.sum(probs)
            G.nodes[goal]['policy'][j] = probs

    best_loss = []
    for start in range(num_nodes):
        G.nodes[start]['tprobs'] = np.zeros(num_nodes)
        for goal in range(num_nodes):
            p = G.nodes[goal]['policy'][start]
            G.nodes[start]['tprobs'] += p

            best_loss.append(-np.sum(p*np.log(p + 1e-10)))
        G.nodes[start]['tprobs'] /= num_nodes

    # generate pairs of nodes from G to be assigned to train or test
    # assigned split is a list of pairs of nodes out of all possible pairs of nodes
    assigned_split = {}
    all_pairs_nodes = list(combinations(G.nodes(),2))
    assigned_split['train'] = random.sample(all_pairs_nodes, int(frac*len(all_pairs_nodes)))
    assigned_split['test'] = [pair for pair in all_pairs_nodes if pair not in assigned_split['train']]
    
    return G, np.mean(best_loss), assigned_split

def compute_policy_degen_modified(G, frac):
    
    num_nodes = G.number_of_nodes()
    G = compute_distance_to_goals(G)

    # G.nodes[list(G.nodes())[i]] = {} by default
    # we will build 3 keys:
    # 'policy' : a num_nodes x num_nodes matrix with the probability of going from node i to node j (goal-conditioned policy)
    # 'tprobs' : a num_nodes x num_nodes matrix with the probability of going from node i to any other node (goal-conditioned marginal policy)
    # 'distance_to_goal' : a num_nodes x num_nodes matrix with the shortest path length from node i to node j

    for goal in G.nodes():
        G.nodes[goal]['policy'] = np.zeros((num_nodes, num_nodes))
        for j in G.nodes():
            probs = np.zeros(num_nodes)
            if j != goal:
                j_neighbors = list(G.neighbors(j))
                valid_moves = 0
                for jn in j_neighbors:
                    '''
                    If the distance to the goal from the neighbor is less than or equal to the distance to the goal from the current node
                    '''
                    if G.nodes[goal]['distance_to_goal'][jn]  <= G.nodes[goal]['distance_to_goal'][j]:
                        probs[jn] = 1
                        valid_moves += 1

                if valid_moves == 0:  # No neighbor is closer to the goal
                    # all neighbors are equally likely
                    probs[j_neighbors] = 1.0 
                    # normalize to have probability 1 since this is a probabilistic policy
                probs /= np.sum(probs)

            else:
                probs[goal] = 1
                
            G.nodes[goal]['policy'][j] = probs

    best_loss = []

    for start in range(num_nodes):
        G.nodes[start]['tprobs'] = np.zeros(num_nodes)
        for goal in range(num_nodes):
            p = G.nodes[goal]['policy'][start]
            G.nodes[start]['tprobs'] += p
            best_loss.append(-np.sum(p*np.log(p + 1e-10)))
        G.nodes[start]['tprobs'] /= num_nodes

        
    # generate pairs of nodes from G to be assigned to train or test
    # assigned split is a list of pairs of nodes out of all possible pairs of nodes
    assigned_split = {}
    all_pairs_nodes = list(permutations(G.nodes(), 2))
    assigned_split['train'] = random.sample(all_pairs_nodes, int(frac*len(all_pairs_nodes)))
    assigned_split['test'] = [pair for pair in all_pairs_nodes if pair not in assigned_split['train']]
    
    return G, np.mean(best_loss), assigned_split


#Generate start-goal-next triplets
def generate_batch(G, assigned_split, batch_size=4096, split='train'):
    '''
    Generate batch_size start-goal-next triplets
    '''

    num_nodes = G.number_of_nodes()


    if split=='test':
        batch_size = len(assigned_split[split])
        goals = np.zeros(batch_size,dtype=int)
        starts = np.zeros(batch_size,dtype=int)
        nexts = np.zeros(batch_size,dtype=int)
        # eval on the whole eval set
        for i in range(len(assigned_split[split])):
            goal,start = assigned_split[split][i]
            goals[i] = goal
            starts[i] = start
            # chose a random next node based on the policy from start towards goal
            nexts[i] = np.random.choice(num_nodes, p=G.nodes[goals[i]]['policy'][starts[i]])
    else:
        goals = np.zeros(batch_size,dtype=int)
        starts = np.zeros(batch_size,dtype=int)
        nexts = np.zeros(batch_size,dtype=int)
        for i in range(batch_size):
            goal,start = random.choice(assigned_split[split])
            goals[i] = goal
            starts[i] = start
            # chose a random next node based on the policy from start towards goal
            nexts[i] = np.random.choice(num_nodes, p=G.nodes[goals[i]]['policy'][starts[i]])

    triplets_raw = np.concatenate((starts[:,None],goals[:,None],nexts[:,None]),axis=1)
    triplets = np.eye(num_nodes)[triplets_raw]
    return triplets, triplets_raw

