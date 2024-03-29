import matplotlib.pyplot as plt 
import networkx as nx
import numpy as np
# import scipy
# from sklearn.metrics import r2_score

def plot_embeddings_adjacencies(Wu, We, V, G):

    start_next=We@Wu
    goal_next=Wu.T@V@We.T

    # Create a figure and a set of subplots: 1 row, 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize to ensure plots are not too cramped

    # Plot 1: $\psi_n^T \phi_s$
    im1 = axs[0].imshow(start_next, aspect="auto", origin="lower")
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title(r"$\psi_n^T \phi_s$")
    axs[0].set_ylabel(r"$n$")
    axs[0].set_xlabel(r"$s$")

    # Plot 2: Adjacency matrix $A$
    im2 = axs[1].imshow(nx.adjacency_matrix(G).toarray(), aspect="auto", origin="lower")
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title(r"$A$")
    axs[1].set_ylabel(r"$n$")
    axs[1].set_xlabel(r"$s$")

    # Plot 3: $\psi_n^T V \phi_g$
    im3 = axs[2].imshow(goal_next, aspect="auto", origin="lower")
    fig.colorbar(im3, ax=axs[2])
    axs[2].set_title(r"$\psi_n^T V \phi_g$")
    axs[2].set_ylabel(r"$n$")
    axs[2].set_xlabel(r"$g$")

    plt.tight_layout()  # Adjust the layout to make sure there's no overlap
    return fig


def plot_neighbour():

    neighbors = []
    nneighbors = []
    for next in range(num_nodes):
        filt = np.array([n for n in G.neighbors(next)])
        arr = np.zeros(num_nodes, dtype = bool)
        arr[filt] = True
        neighbors += list(start_next[next,filt])
        nneighbors += list(start_next[next,~filt])    

    plt.close("all")
    plt.hist(neighbors, bins = 50, alpha = 0.5, label = "neighbors")
    plt.hist(nneighbors, bins = 50, alpha = 0.5, label = "non-neighbors")
    plt.legend()
    plt.ylabel("Count")
    plt.xlabel(r"$\psi_n^T \phi_s$")
    plt.tight_layout()
    plt.savefig("figs/neighbors_hist_N%d_p%.2f_Ne%d_m%d_G" %(num_nodes,p,Ne,method) + graph_type + ".png")
    plt.show()

    neighbors_mean = np.mean(neighbors)
    nneighbors_mean = np.mean(nneighbors)

    print(neighbors_mean, nneighbors_mean)

def plot_and_fit_short_bias():

    fig,axis = plt.subplots(1,1,figsize=(5,5))
    xs = []
    ys = []
    for next in range(num_nodes):
        distances = nx.shortest_path_length(G,source=next)
        dists = np.array([distances[i] for i in range(num_nodes)])
        xs.append(list(dists))
        ys.append(list(goal_next[next,:]))

    xs = np.array(xs)
    ys = np.array(ys)
    ys_means = []
    axis.plot(xs,ys,"ko",alpha=0.5)
    for i in range(np.max(xs)+1):
        filt = xs == i
        axis.plot(i,np.mean(ys[filt]),"rs")
        ys_means += [np.mean(ys[filt])]

    ps = np.polyfit(np.arange(np.max(xs)+1),np.array(ys_means),1)
    r2 = r2_score(np.array(ys_means),ps[0]*np.arange(np.max(xs)+1)+ps[1])
    print(r2)
    axis.plot(np.arange(np.max(xs)+1), ps[0]*np.arange(np.max(xs)+1)+ps[1],"r--")
    axis.set_xlabel("Distance")
    axis.set_ylabel(r"$\psi_n^T V \phi_g$")
    axis.set_title(r"$R^2$ = %.3f"%r2)
    fig.tight_layout()
    fig.savefig("figs/shortest_path_N%d_p%.2f_Ne%d_m%d_G" %(num_nodes,p,Ne,method) + graph_type + ".png")
    plt.show()
