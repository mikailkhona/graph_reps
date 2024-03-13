import networkx as nx
import numpy as np

def compute_distance_to_goals(G):
    for goal in G.nodes():
        G.nodes[goal]['distance_to_goal'] = np.zeros(G.number_of_nodes())
        for node in G.nodes():
            G.nodes[goal]['distance_to_goal'][node] = nx.shortest_path_length(G, node, goal)
    return G

def compute_policy_degen(G):
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
    
    return G,np.mean(best_loss)

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

#Generate start-goal-next triplets
def generate_batch(G,batch_size):
    '''
    Generate batch_size start-goal-next triplets
    '''

    num_nodes = G.number_of_nodes()
    goals = np.zeros(batch_size,dtype=int)
    starts = np.zeros(batch_size,dtype=int)
    nexts = np.zeros(batch_size,dtype=int)

    for i in range(batch_size):
        goal,start = np.random.choice(num_nodes,2,replace=False)
        goals[i] = goal
        starts[i] = start
        nexts[i] = np.random.choice(num_nodes, p=G.nodes[goals[i]]['policy'][starts[i]])

    triplets_raw = np.concatenate((starts[:,None],goals[:,None],nexts[:,None]),axis=1)
    triplets = np.eye(num_nodes)[triplets_raw]
    return triplets, triplets_raw
