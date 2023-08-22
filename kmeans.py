'''kmeans.py
Performs K-Means clustering
Junnan Shimizu
CS 252: Mathematical Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        self.data = data
        self.num_samps = np.shape(data)[0]
        self.num_features = np.shape(data)[1]

        pass

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''

        return np.copy(self.data)

        pass

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        dist = np.sqrt(np.sum(np.square(pt_1 - pt_2)))

        return dist

        pass

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        dist = np.sqrt(np.sum(np.square(centroids - pt), axis=1))

        return dist

        pass

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''

        # indices = np.random.randint(0, np.shape(self.data)[0], size=k)
        indices = np.random.choice(np.arange(0, np.shape(self.data)[0]), size=k, replace=False)
        self.centroids = self.data[indices]

        return self.centroids

        pass

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''

        centroids = [self.data[np.random.randint(0, np.shape(self.data)[0])]]

        min_distances = np.zeros(self.num_samps)

        for i in range(self.num_samps):
            min_distances[i] = np.min(self.dist_pt_to_centroids(self.data[i], centroids))

        for i in range(1, k):
            probabilities = np.divide(np.square(min_distances), np.sum(np.square(min_distances)))
            cdf = np.cumsum(probabilities)

            r = np.random.uniform()
            index = np.searchsorted(cdf, r)

            centroids.append(self.data[index])

        self.centroids = np.array(centroids)

        return self.centroids

        pass

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, init_method='random'):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''

        self.k = k

        if init_method == 'random':
            self.initialize(self.k)
        else:
            self.initialize_plusplus(self.k)

        self.num_samps = np.shape(self.data)[0]
        self.num_features = np.shape(self.data)[1]

        self.centroids, centroid_dif = self.update_centroids(k, self.data_centroid_labels, self.centroids)
        self.data_centroid_labels = self.update_labels(self.centroids)

        count = 0
        while count < max_iter and np.abs(centroid_dif).any() > tol:
            self.update_labels(self.centroids)
            self.centroids, centroid_dif = self.update_centroids(k, self.data_centroid_labels, self.centroids)
            count += 1
            if centroid_dif.all() == 0:
                break

        self.inertia = self.compute_inertia()

        return self.inertia


        pass

    def cluster_batch(self, k=2, n_iter=1, verbose=False, init_method='random'):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''

        min_inertia = 9999999
        total_iters = 0
        set_total = 0

        for test_k in range(k):
            for i in range(n_iter):
                inertia = self.cluster(test_k + 1, init_method='cluster')
                total_iters += 1
                if inertia < min_inertia:
                    min_inertia = inertia
                    temp_centroids = self.centroids
                    temp_labels = self.data_centroid_labels
                    set_total = total_iters

        self.centroids = temp_centroids
        self.data_centroid_labels = temp_labels
        self.inertia = min_inertia

        return set_total / n_iter

        pass

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''

        nearest_centroids = []

        for index in range(self.num_samps):
            nearest_centroids.append(np.argmin(self.dist_pt_to_centroids(self.data[index, :], centroids)))

        return np.array(nearest_centroids)

        pass

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''

        # self.k = k
        new_centroids = np.zeros(shape=(np.shape(prev_centroids)))
        sum = np.zeros((0, np.shape(self.centroids)[1]))

        for i in range(k):
            indices = np.where(data_centroid_labels==i)
            if np.shape(indices)[1] == 0:
                new_centroids[i] = self.data[np.random.randint(k)]
                continue
            sum = np.sum(self.data[indices, :], axis=1)
            new_centroids[i] = (sum)/(np.shape(indices)[1])

        centroid_dif = new_centroids - prev_centroids  

        return new_centroids, centroid_dif
                
        pass

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''

        mean_sqr_dist = 0

        for k in range(len(self.centroids)):
            indices = np.where(self.data_centroid_labels==k)
            mean_sqr_dist += np.square(np.sqrt(np.sum(np.square(np.subtract(self.centroids[k], self.data[indices])))))

        mean_sqr_dist = mean_sqr_dist / np.shape(self.data)[0]

        return mean_sqr_dist

        pass

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''

        # print(self.data_centroid_labels)
        cmap = plt.get_cmap('Set1')

        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data_centroid_labels, cmap=cmap)
        plt.scatter(x=self.centroids[:, 0], y=self.centroids[:, 1], marker='P')

        pass

    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: int. Number of times to run K-means with the designated `k` value.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        
        cmap = plt.get_cmap('Set1')

        inertia_list = np.zeros(shape=(max_k, 1))

        # for k in range(max_k):
        #     inertia_list[k] = self.cluster(k + 1)

        for k in range(max_k):
            self.cluster_batch(k=k + 1, n_iter=n_iter)
            inertia_list[k] = self.inertia

        ax = plt.axes()
        plt.plot(range(1, max_k + 1), inertia_list)
        plt.xlabel('k clusters')   
        plt.ylabel('Inertia')
        ax.set_xticks(range(1, max_k))        
        # ax.set_yticks(range(max(inertia_list)))
       

        pass

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''

        data = self.get_data()
        data[:, 1] = self.data_centroid_labels
        self.set_data(data)

        pass
