'''
coding:UTF-8
Assignment 2-CIS694/EEC693/CIS593 Deep Learning-2022 Spring
k-means clustering scratch
author: sabareeswaran shanmugam
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Task :1
################# 1 Implement your own kmeans clustering function myKmeans() (60 points) #################
# Write your code here for your own kmeans clustering function
def myKmeans(X, k, max_iteration):
    '''write your code here based on the algorithm described in class, and you cannot call other kmeans clustering packages here
    '''
    idx = np.random.choice(len(X), k, replace=False)
    # Step 1:Randomly choosing Centroids
    centroids = X[idx, :]
    # Step 2: finding the distance between centroids and all the data points
    intra_cluster_distances = cdist(X, centroids, 'euclidean')
    # Step 3: Update the Centroid with the minimum Distance
    center = np.array([np.argmin(i) for i in intra_cluster_distances])

    # Step 4: Repeating the above steps for a defined number of iterations

    for _ in range(max_iteration):
        centroids = []

        for idx in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp = X[center == idx].mean(axis=0)
            centroids.append(temp)

        updated_centroid = np.vstack(centroids)  # Updated Centroids

        # finding intra_distance between updated centroids and all the data points
        intra_cluster_distances = cdist(X, updated_centroid, 'euclidean')
        # finding inter_distance between two updated centroids
        inter_cluster_distance = cdist(updated_centroid, updated_centroid, 'euclidean')

        center = np.array([np.argmin(i) for i in intra_cluster_distances])
        center2 =np.array([np.argmin(i) for i in inter_cluster_distance])
        # calculating the mean of both intra and inter distance of cluster.
        mean_intra_cluster_distance=center.mean(axis=None)
        mean_inter_cluster_distance=center2.mean(axis=None)
        
    return center,mean_intra_cluster_distance,mean_inter_cluster_distance

################# 2 Optimal K for your own kmeans clustering (30 points) #################

# Write your code for a loop to call your own function myKmeans() by setting cluster_number=K from 2 to 10
# print the ratio of mean_intra_cluster_distance over mean_inter_cluster_distance for each K.
# print the optimal K with minimum ratio
def optimal_K_find(fake_data):
    #k =[2,3,4,5,6,7,8,9,10]
    ratio_array = []
    for i in range(2,11):
        # here calling own-mykmeans function for different values of k
        labels, intra, inter = myKmeans(fake_data,i, 1000)
        r = (float("{0:.1f}".format(intra))/ float("{0:.1f}".format(inter)))
        ratio_rounded =round(r,2)
        ratio_array.append(ratio_rounded)
        print("when k=" + str(i) +",Mean intra-cluster distance:" + str(float("{0:.1f}".format(intra))) + ",Mean inter-cluster distance:" + str(float("{0:.1f}".format(inter))) + ",Ratio :" + str(r))
    #print(ratio_array)
    optimal_ratio =min(ratio_array)
    # here appending 2 because list index starts from 0
    optimal_k = ratio_array.index(optimal_ratio) +2
    print("The optimal k =" + str(optimal_k) +",because k =" + str(optimal_k) +" obtains the minimum ratio "+ str(optimal_ratio))



if __name__ == '__main__':
    # make fake data by normal distribution (mean, std)
    n_data = torch.ones(1000, 2)
    x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    x2 = torch.normal(-5 * n_data, 1)  # class2 x data (tensor), shape=(100, 2)
    x3 = torch.normal(5 * n_data, 1)  # class3 x data (tensor), shape=(100, 2)
    x4 = torch.normal(10 * n_data, 1)  # class4 x data (tensor), shape=(100, 2)
    x = torch.cat((x0, x1, x2, x3, x4), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    print("The fake tensor data:\n",x)

    # Task 2
    print("Executing Task 2 calling optimal_k_find function and setting cluster_number=K from 2 to 10")
    ################# 2 Optimal K for your own kmeans clustering (30 points) #################
    # calling optimal_k_find function and passing k from 2 to 10
    optimal_K_find(x)

    # Task 3
    print("Executing Task 3 Calling my own function myKmeans() by setting K=5 for visualization")
    ################# 3 Call your own function myKmeans() by setting K=5 for visualization (10 points) #################
    # Write your code to call your own function myKmeans() by K=5, and visulize the cluster results, which should be similar to the first figure
    labels, intra, inter = myKmeans(x, 5, 1000)
    cluster_labels =np.unique(labels)
    for i in cluster_labels:
        plt.scatter(x[labels == i, 0], x[labels == i, 1], label=i)
    plt.legend()
    plt.show()



"""
Reference :
1) https://www.askpython.com/python/examples/k-means-clustering-from-scratch
2)https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
3)https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

"""