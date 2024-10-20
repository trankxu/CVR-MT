import torch
import torch.nn
from torch.nn import functional as F
import numpy as np
import random
import copy
"""
The different uncertainty methods loss implementation.
Including:
    Ignore, Zeros, Ones, SelfTrained, MultiClass
"""

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [1113, 6705, 514, 327, 1099, 115, 142]
CLASS_WEIGHT = torch.Tensor([10000/i for i in CLASS_NUM]).cuda()

class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output_softmax, target.long())

##############reference from https://blog.csdn.net/mch2869253130/article/details/119752937
class K_means():
    def __init__(self, data, k):
        self.data = data
        self.k = k
        

    def distance(self, p1, p2):
        return torch.sum((p1-p2)**2).sqrt()

    def generate_center(self):
        # 随机初始化聚类中心
        n = self.data.size(0)
        #rand_id = random.sample(range(n), self.k)
        
        rand_id = torch.randint(self.data.size(dim = 0),(self.k,))
        center = self.data[rand_id,:].clone()
        #for id in rand_id:
            #center.append(self.data[id])
        return center

    def converge(self, old_center, new_center):
        # 判断是否收敛
        diff = torch.sum(torch.norm(old_center - new_center, 2, 1))
        #print("old_center",old_center, "new_center",new_center)
        #print(diff)
        return torch.isclose(diff,torch.zeros_like(diff))

    def forward(self):
        center = self.generate_center()
        n = self.data.size(0)
        labels = torch.zeros(n).long()
        diff = False
        #print("one step")
        while diff != torch.tensor([True]).cuda():
            old_center = copy.deepcopy(center.detach()) #需要梯度 所以无法使用deepcopy 需要detach

            for i in range(n):
                cur = self.data[i]
                min_dis = 10*9
                for j in range(self.k):
                    dis = self.distance(cur, center[j])
                    if dis < min_dis:
                        min_dis = dis
                        labels[i] = j

            # 更新聚类中心
            for j in range(self.k):
                center[j] = torch.mean(self.data[labels == j], dim=0)
                if torch.isnan(center[j]).any() == torch.tensor([True]).cuda():
                    center[j] = torch.isnan(center[j]).int()
                #print((labels == j).shape)
                #print(self.data[labels == j].shape)
            
            diff = self.converge(old_center, center)
            

        return labels, center

# class weighted_cross_entropy_loss(object):
#     """
#     map all uncertainty values to a unique value "2"
#     """
    
#     def __init__(self):
#         self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
#     def __call__(self, output, target):
#         # target[target == -1] = 2
#         output_softmax = F.softmax(output, dim=1)
#         target = torch.argmax(target, dim=1)
#         return self.base_loss(output_softmax, target.long())

def get_UncertaintyLoss(method):
    assert method in METHODS
    
    if method == 'U-Zeros':
        return Loss_Zeros()

    if method == 'U-Ones':
        return Loss_Ones()
    
    if method == 'U-MultiClass':
        return Loss_MultiClass()

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).to("cuda:3")
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    return mse_loss

def cam_attention_map(activations, channel_weight):
    # activations 48*49*1024
    # channel_weight 48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    attention = attention.permute(1,0,2)
    attention = torch.sum(attention, -1)
    attention = torch.reshape(attention, (48, 7, 7))

    return attention

def cam_activation(batch_feature, channel_weight):
    # batch_feature = batch_feature.permute(0,2,3,1)#48 7 7 1024
    # activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))#48*49*1024
    
    # attention = activations.permute(1,0,2)#.mul(channel_weight)#49*48*1024
    # attention = attention.permute(1,2,0)#48*1024*49
    # attention = F.softmax(attention, -1)#48*1024*49

    # activations2 = activations.permute(0, 2, 1) #48 1024 49
    # activations2 = activations2 * attention 
    # activations2 = torch.sum(activations2, -1)#48*1024
    batch_feature = batch_feature.permute(0,2,3,1)
    #48*49*1024
    activations = torch.reshape(batch_feature, (batch_feature.shape[0], -1, batch_feature.shape[3]))
    
    #49*48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    #48*49*1024
    attention = attention.permute(1,0,2)
    #48*49
    attention = torch.sum(attention, -1)
    attention = F.softmax(attention, -1)

    activations2 = activations.permute(2, 0, 1) #1024*48*49
    activations2 = activations2 * attention 
    activations2 = torch.sum(activations2, -1) #1024*48
    #48 1024 
    activations2 = activations2.permute(1,0)

    return activations2

# def relation_mse_loss_cam(activations, ema_activations, model, label):
#     """Takes softmax on both sides and returns MSE loss

#     Note:
#     - Returns the sum over all examples. Divide by the batch size afterwards
#       if you want the mean.
#     - Sends gradients to inputs but not the targets.
#     """
#     weight = model.module.densenet121.classifier[0].weight
#     #48*1024
#     channel_weight = label.mm(weight)

#     activations = cam_activation(activations.clone(), channel_weight)
#     ema_activations = cam_activation(ema_activations.clone(), channel_weight)
    
#     assert activations.size() == ema_activations.size()

#     activations = torch.reshape(activations, (activations.shape[0], -1))
#     ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

#     similarity = activations.mm(activations.t())
#     norm_similarity = similarity / torch.norm(similarity, p=2)

#     ema_similarity = ema_activations.mm(ema_activations.t())
#     norm_ema_similarity = ema_similarity / torch.norm(ema_similarity, p=2)

#     similarity_mse_loss = (norm_similarity-norm_ema_similarity)**2
#     return similarity_mse_loss


def relation_mse_loss_cam(activations, ema_activations, model, label):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    weight = model.module.densenet121.classifier[0].weight
    #48*1024
    channel_weight = label.mm(weight)

    activations = cam_activation(activations.clone(), channel_weight)
    ema_activations = cam_activation(ema_activations.clone(), channel_weight)
    
    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss

def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()
    #print("activatetion:",activations.shape,ema_activations.shape)
    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss


def feature_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    # similarity = activations.mm(activations.t())
    # norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    # norm_similarity = similarity / norm

    # ema_similarity = ema_activations.mm(ema_activations.t())
    # ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    # ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (activations-ema_activations)**2
    return similarity_mse_loss


def sigmoid_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    mse_loss = loss_fn(input_softmax, target_softmax)
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def KMeans(x, output, ema_output, K=10, Niter=500, init_cent = None, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    import time
    import collections
    use_cuda = torch.cuda.is_available()
    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    #c = x[:K, :].clone()  # Simplistic initialization for the centroids
    #torch.manual_seed(stream_class)
    #init_c = c
    
    ########Plan2
    if init_cent is not None:
        #print(init_cent)
        init_c = init_cent[0]

        
        for key in init_cent:
            if key != 0:
                init_c = torch.vstack((init_c, init_cent[key]))
        
        c = init_c
        if c.grad_fn is not None:
            print('c is in computation')
    else:
                
        slice_ = torch.randint(x.size(dim = 0),(K,))
        init_c = x[slice_, :].clone()
        c = init_c
        
    ################
    
    ########Plan3
    #torch.cuda.manual_seed(stream_class)
    #print(x.size(dim = 1))
    #c = torch.rand(K, x.size(dim = 1)).to("cuda:0")
    #################################################
    from sklearn.manifold import TSNE
    from umap import UMAP
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    import warnings
    # ignore all future warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        tsne = TSNE(n_components=3)
        umap = UMAP(n_neighbors = 10, n_components = 3, n_epochs = 80)
        projected_tensors_data = umap.fit_transform(x.clone().detach().cpu())

        projected_center = umap.fit_transform(c.detach().cpu())
    
    x_i = torch.tensor(projected_tensors_data).view(N, 1, 3).to("cuda:0")  # (N, 1, D) samples
    c_j = torch.tensor(projected_center).view(1, K, 3).to("cuda:0")  # (1, K, D) centroids
    '''
    for i in range(48):
        close_point = 0
        close_dis = np.inf
        for j in range(7):
            dis = ((projected_tensors_data[i]-projected_center[j])**2).sum()
            if dis < close_dis:
                close_dis = dis
                close_point = j

        print("The closest centroid is:", close_point)
    '''
    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # (N,)Points -> Nearest cluster
        
        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1) #(k,1) 统计每个类的频数
        #print("number of points in each cluster:",Ncl)
        c /= Ncl + 1e-15  # in-place division to compute the average
    

    
    '''
    # Plot the resulting tensors using scatter plots
    plt.scatter(projected_tensors_data[:, 0],projected_tensors_data[:,1],projected_tensors_data[:,2])
    plt.scatter(projected_center[:,0],projected_center[:,1],projected_center[:,2], c = 'y')
    plt.show()
    '''
    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c ,init_c

def cluster_consistency_loss(activations, ema_activations, output, ema_output, init_center, k = 3):
    """Calculates the consistency loss based on KMeans of images.

    Parameters:
    - activations: A 2D array of shape (n_samples, n_features) containing the input images.
    - ema_activations: A 2D array of shape (n_clusters, n_features) containing the cluster centers
      predicted by the model at the previous time step.

    Returns:
    - loss: A float value representing the consistency loss.
    """
    # Create a k-means model

    

    use_cuda = torch.cuda.is_available()
    #dtype = torch.float32 if use_cuda else torch.float64
    #device_id = "cuda:0" if use_cuda else "cpu"
    
    #print("activation:",ema_activations, activations)
    #print(output.shape, output)
    
    _, ema_centers, _ = KMeans(ema_activations, output, ema_output, k, init_cent = init_center, verbose= False) # ema input 
    _, origin_centers, _ = KMeans(activations, output, ema_output, k, init_cent = init_center, verbose= False) #original input
    
    #print("centers:", ema_centers,origin_centers)
    '''
    kmeans_input = K_means(activations , 7)
    kmeans_ema = K_means(ema_activations, 7)
    
    labels_input, centers_orig = kmeans_input.forward()
    _, centers_ema = kmeans_input.forward()
    #print(centers_orig, centers_ema)
    loss = 
    torch.norm(centers_orig - centers_ema, 2, 1)
    '''
    '''   
    kmeans_input = KMEANS(n_clusters = k,max_iter = 1000, verbose = False, device = torch.device("cuda:0"))
    kmeans_ema = KMEANS(n_clusters = k,max_iter = 1000, verbose = False, device = torch.device("cuda:0"))
    
    kmeans_input.fit(activations)
    kmeans_ema.fit(ema_activations)
    # Get the cluster centers predicted by the model for the current time step
    origin_centers = kmeans_input.centers#Kmeans_cluster(activations, k,100)
    ema_centers =  kmeans_ema.centers#(ema_activations
    #print(origin_centers,ema_centers)
    # Calculate the average distance between the current and previous cluster centers
    '''
    loss = torch.norm(origin_centers - ema_centers, 2, 1)
    
    #print(torch.norm(origin_centers - ema_centers, 2, 1).shape)
    return loss

