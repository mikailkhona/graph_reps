import torch
import torch.nn as nn
import torch.nn.functional as F

def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)

class NTXEntLoss(nn.Module):
    def _init__(self, batch_size, temperature=0.5)-> None:
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size*2, batch_size*2,dtype=bool)).float()

    def calc_similarity(self, a, b):
        representations = torch.cat([a,b],dim=0)
        return F.cosine_similarity(representations.unsqueeze(0), representations.unsqueeze(1), dim=2)
    
    def forward(self, proj_1, proj_2):
        '''
        Inputs: proj_1, proj_2 (2 views)
        proj_1 and proj_2 are batch embeddings (batch_size, embedding_dimensionality)
        where corresponding indices are pairs z_i and z_j in the simCLR paper notation
        '''
        
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2,dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        # N = batch_size
        # Generate 2 views for each element in the batch using augmnetations
        # for i \in {1, 2, ..., 2N} and j \in {1, 2, ...., 2N}
        # Calculate the similarity matrix using the following
        # sim_{i,j} = cosine_similarity(z_i, z_j)

        similarity_matrix = self.calc_similarity(z_i,z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, - batch_size)

        positives = torch.cat([sim_ij, sim_ji],dim=0)

        numerator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        
        loss = torch.sum(all_losses) / (2 * self.batch_size)

        return loss


def contrastiveLoss(preds, labels):   
        softmaxed_preds = F.softmax(preds, dim=-1)

        # argmax'ed labels
        label_idxs = torch.argmax(labels,dim=1)

        #random labels
        random_idxs = torch.randint(low=0, high=softmaxed_preds.shape[1],size=(softmaxed_preds.shape[0],))

        loss = torch.log(softmaxed_preds[torch.arange(softmaxed_preds.shape[0]),label_idxs]) + torch.log(1-(softmaxed_preds[torch.arange(softmaxed_preds.shape[0]), random_idxs]))

        return loss


# def align_unif(log_lambda, real, noise, batch_size):

# #       phi, _psi = repr_fn.apply(params, x0)
# #   _phi, psi = repr_fn.apply(params, xT)
    
#     # put representations on the sphere
#     normalized_real = F.normalize(real, dim=)
#     normalized_noise = F.normalize(noise, dim=)

#     l2 = (torch.mean(real**2) + torch.mean(noise**2)) / 2
#     I = torch.eye(batch_size)
#     l_align = torch.sum((phi - psi)**2, dim=1)

#     pdist = torch.mean((phi[:, None] - psi[None])**2, dim=-1)
#     l_unif = (torch.logsumexp(-(pdist * (1 - I)), dim=1) + torch.logsumexp(-(pdist.T * (1 - I)), dim=1)) / 2.0

#     loss = l_align + l_unif

#     accuracy = torch.mean(torch.argmin(pdist, dim=1) == torch.arange(batch_size))
#     dual_loss = log_lambda * (c - l2.detach())
#     metrics = (l_unif.mean(), l_align.mean(), accuracy, l2)
#     return loss.mean() + torch.exp(log_lambda) * l2 + dual_loss, metrics

# class CLIPLoss(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:

#     def get_ground_truth(self, ):

#         labels = torch.arange()

#     def forward(self, features_modality1, features_modality2):
#         labels = 