'''rbf_net.py
Radial Basis Function Neural Network
Junnan Shimizu
CS 252: Mathematical Data Analysis Visualization
Spring 2023
'''
import numpy as np
import kmeans

import scipy.linalg

class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''

        # number of hidden units as an instance variable
        self.k = num_hidden_units

        # number of classes (number of output units in network)
        self.num_classes = num_classes

        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''

        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''

        return self.k

        pass

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''

        return self.num_classes

        pass

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''

        k = np.shape(centroids)[0]
        distances = np.zeros(shape=(k))

        for cluster in range(k):
            indices = np.where(cluster_assignments == cluster)
            cluster_data = data[indices]
            distances[cluster] = np.mean(kmeans_obj.dist_pt_to_centroids(cluster_data, centroids[cluster, :]))

        return distances

        pass

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''

        kmean = kmeans.KMeans(data)
        kmean.initialize_plusplus(self.num_classes)
        kmean.cluster_batch(k=self.num_classes, n_iter=5)

        self.prototypes = kmean.get_centroids()
        self.sigmas = self.avg_cluster_dist(data, kmean.get_centroids(), kmean.get_data_centroid_labels(), kmean)

        pass

    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''

        A = np.hstack([A, np.ones((np.shape(A)[0], 1))])
        Q, R = self.qr_decomposition(A)

        Qty = np.dot(np.transpose(Q), y)
        c = scipy.linalg.solve_triangular(R, Qty)

        num_samples = np.shape(c)[0]

        self.slope = c[:num_samples - 1, :1]
        self.intercept = c[1]

        return c

        pass

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''

        Q = np.zeros(shape=np.shape(A))

        for i in range(np.shape(A)[1]):
            j = A[:, i]
            for i2 in range(i):
                q = Q[:, i2]
                j = j - np.dot(q, j) * q
            Q[:, i] = j / np.linalg.norm(j)

        R = np.dot(np.transpose(Q), A)

        return Q, R

    def hidden_act(self, data):
        '''Cn

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''

        hidden_act = np.zeros((np.shape(data)[0], np.shape(self.prototypes)[0]))

        for i, sample in enumerate(data):
            for j, center in enumerate(self.prototypes):
                dist = self.dist_pt_to_pt(sample, center)
                hidden_act[i, j] = np.exp((-1 * np.square(dist)) / (2 * np.square(self.sigmas[j]) + 1e-8))

        return hidden_act

        pass

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''

        new_row = np.ones(shape=(np.shape(hidden_acts)[0], 1))

        hidden_acts = np.hstack((hidden_acts, new_row))

        return hidden_acts @ self.wts

        pass

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''

        self.initialize(data)

        recoded_y = np.zeros((np.shape(y)[0], self.get_num_hidden_units()))

        for i, y_val in enumerate(y):
            recoded_y[i, y_val] = 1

        hidden_act = self.hidden_act(data)

        self.wts = self.linear_regression(hidden_act, recoded_y)

        pass

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''

        hidden_act = self.hidden_act(data)

        output_act = self.output_act(hidden_act)

        predicted_classes = np.argmax(output_act, axis=1)

        return predicted_classes

        pass

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''

        return np.mean(y == y_pred)

        pass
    
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


class RBF_Reg_Net(RBF_Net):
    '''RBF Neural Network configured to perform regression
    '''
    def __init__(self, num_hidden_units, num_classes, h_sigma_gain=5):
        '''RBF regression network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        h_sigma_gain: float. Multiplicative gain factor applied to the hidden unit variances

        TODO:
        - Create an instance variable for the hidden unit variance gain
        '''
        super().__init__(num_hidden_units, num_classes)

        self.h_sigma_gain = h_sigma_gain

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation..
            See notebook for refresher on the activation equation

        TODO:
        - Copy-and-paste your classification network code here.
        - Modify your code to apply the hidden unit variance gain to each hidden unit variance.
        '''

        hidden_act = np.zeros((np.shape(data)[0], np.shape(self.prototypes)[0]))

        for i, sample in enumerate(data):
            for j, center in enumerate(self.prototypes):
                dist = self.dist_pt_to_pt(sample, center)
                hidden_act[i, j] = np.exp((-1 * np.square(dist)) / (2 * self.h_sigma_gain * np.square(self.sigmas[j]) + 1e-8))

        return hidden_act

        pass

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the desired y output of each training sample.

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to perform regression on
        the actual y values instead of the y values that match a particular class. Your code should be
        simpler than before.
        - You may need to squeeze the output of your linear regression method if you get shape errors.
        '''

        self.initialize(data)

        hidden_act = self.hidden_act(data)

        # y = np.squeeze(y, axis=1)

        self.wts = self.linear_regression(hidden_act, y)

        pass

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_neurons). Output layer neuronPredicted "y" value of
            each sample in `data`.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to return the RAW
        output neuron activaion values. Your code should be simpler than before.
        '''

        hidden_act = self.hidden_act(data)

        output_act = self.output_act(hidden_act)

        # predicted_classes = np.argmax(output_act, axis=1)

        return output_act

        pass

    
