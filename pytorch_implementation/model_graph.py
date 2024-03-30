
import numpy as np
import torch 
import torch.nn.functional as F
import torch.nn as nn
import pdb

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