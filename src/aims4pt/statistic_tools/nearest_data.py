import pandas as pd
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class my_dataset_cluster:
    """
    A DIY class for clustering datasets using hierarchical clustering.

    With a automatic determination of the number of clusters using the Silhouette score.
    """
    def __init__(self, X: pd.DataFrame, additional_info: Optional[pd.DataFrame|pd.Series] = None, k: Optional[int] = None):
        """
        Initialize the clustering with the dataset and optional number of clusters.

        Parameters
        ----------
        X : pd.DataFrame
            The original dataset to be clustered.
        additional_info : Optional[pd.DataFrame|pd.Series], default=None
            Additional information to be concatenated with the dataset. This can be used to include target variables
        k : int or None, optional
            The number of clusters to form. If None, the number of clusters will be determined automatically through the Silhouette score.
        """
        self.X = X
        self.additional_info = additional_info
        self.k = k
        self.full_dataset = X.copy()
        if additional_info is not None:
            self.full_dataset = pd.concat([self.full_dataset, self.additional_info], axis=1)

    def fit(self):
        """
        Fit the clustering model to the dataset.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        max_k = np.sqrt(self.X.shape[0]).astype(int)

        if self.k is None:
            # Automatically determine the number of clusters using the Silhouette score
            best_k = 2
            best_score = -1
            for k in range(2,max_k):
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(self.X)
                score = calinski_harabasz_score(self.X, labels)
                if score > best_score:
                    best_k = k
                    best_score = score
            self.k = best_k

        # plot score and k
        print(f"Optimal number of clusters determined: {self.k}")
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_k), [calinski_harabasz_score(self.X, AgglomerativeClustering(n_clusters=k).fit_predict(self.X)) for k in range(2, max_k)], marker='o')
        plt.title('Calinski-Harabasz Score vs Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Calinski-Harabasz Score')
        plt.axvline(x=self.k, color='r', linestyle='--')
        plt.show()

        model = AgglomerativeClustering(n_clusters=self.k)
        self.labels_ = model.fit_predict(self.X)
        return self


    def get_labels(self):
        """
        Get the cluster labels for the dataset.

        Returns
        -------
        pd.Series
            A Series containing the cluster labels for each row in the dataset.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        return pd.Series(self.labels_, index=self.X.index, name='Cluster')
    
    def get_clustered_dataset(self):
        """
        Get the clustered dataset with an additional column for cluster labels.

        Returns
        -------
        pd.DataFrame
            The original dataset with an additional 'Cluster' column indicating the cluster each row belongs to.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        clustered_data = self.full_dataset.copy()
        clustered_data['Cluster'] = self.labels_
        return clustered_data
    
    def get_cluster_centers(self):
        """
        Get the cluster centers for the clustered dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the mean of each feature for each cluster.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        return self.full_dataset.groupby(self.labels_).mean()

    def get_cluster_sizes(self):
        """
        Get the sizes of each cluster.
        Returns
        -------
        pd.Series
            A Series containing the size of each cluster.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        return self.full_dataset.groupby(self.labels_).size()

    def predict(self, X_new: pd.DataFrame):
        """
        Predict the cluster labels for new data.

        Parameters
        ----------
        X_new : pd.DataFrame
            The new data to predict cluster labels for.

        Returns
        -------
        pd.Series
            A Series containing the predicted cluster labels for each row in the new data.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        
        from sklearn.neighbors import NearestCentroid
        model = NearestCentroid()
        model.fit(self.X, self.labels_)
        return pd.Series(model.predict(X_new), index=X_new.index, name='Predicted Cluster')
    
    def get_neighbors(self, X_new: pd.DataFrame):
        """
        Get the original data corresponding to the predicted cluster labels for new data.

        Parameters
        ----------
        X_new : pd.DataFrame
            The new data to get original data for.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the original data for each row in the new data.
        """
        predicted_labels = self.predict(X_new)
        labels = predicted_labels.unique()
        original_data = pd.DataFrame()
        for label in labels:
            original_data = pd.concat([original_data, self.full_dataset[self.labels_ == label]], ignore_index=True)
        return original_data
    
    


class MyDatasetNeighbors_radius:
    """
    A DIY class for neighborhood-based retrieval using radius or k-NN,
    with automatic radius determination.
    """
    def __init__(
        self,
        X: pd.DataFrame,
        weight: pd.DataFrame = None,
        additional_info: Optional[pd.DataFrame | pd.Series] = None,
        radius: Optional[float] = None,
        percentile: float = 10.0,

    ):
        """
        Initialize with features, predicted labels, and optional settings.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for training samples.
        weight : pd.DataFrame, optional
            Weight matrix for training samples, if applicable.
        additional_info : DataFrame or Series, optional
            Extra columns to keep with the dataset (e.g. metadata).
        radius : float or None, default=None
            Neighborhood radius. If None, will be set as the `percentile`-th
            percentile of the k-th neighbor distances.
        percentile : float in (0,100), default=10.0
            Which percentile of the k-th neighbor distances to pick as radius.

        """
        self.X = X
        self.weight = weight
        self.additional_info = additional_info
        self.radius = radius
        self.percentile = percentile
        self.scaler = None


        # build full dataset for lookup
        self.full_df = X.copy()
        if additional_info is not None:
            self.full_df = pd.concat([self.full_df, additional_info], axis=1)

    def process_X(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_temp = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_temp, columns=self.X.columns, index=self.full_df.index)
        if self.weight is not None:
            self.weights_array = np.array([self.weight.get(f, 1.0)
                           for f in self.X.columns], dtype=float)
            self.weights_array = np.sqrt(self.weights_array)
            self.X = self.X.mul(self.weights_array, axis=1)

    def fit(self):
        """
        Determine radius (if needed) and fit the NearestNeighbors model.
        Also plots the k-th neighbor distance distribution with percentile cutoff.
        """
        np.random.seed(42)  # for reproducibility
        self.process_X()


        # 1) estimate radius if not provided
    
        if self.radius is None:
            from sklearn.metrics import pairwise_distances
            D_tt = pairwise_distances(self.X, metric='euclidean')
            # Use only the upper triangle to avoid self-distances of zero.
            i,j = np.triu_indices_from(D_tt, k=1)
            d_train = D_tt[i,j]
            print("train–train distance:", np.percentile(d_train, [10,50,90]))

            # 3) Percentile.
            radius = np.percentile(d_train, self.percentile)
            self.radius = radius


        # 2) fit radius-based NN
        self.nn_radius = NearestNeighbors(
            radius=self.radius, metric='euclidean'
        ).fit(self.X)



        return self



    def get_neighbors(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve the original training samples that fall in each neighborhood.

        Returns
        -------
        pd.DataFrame
            Concatenated neighbors for all X_new, with an extra column
            'query_index' to indicate which test sample they belong to.
        """
        dfs = []
        if self.scaler is not None:
            # If a scaler is available, standardize the new data first.
            X_new = self.scaler.transform(X_new)
            X_new = pd.DataFrame(X_new, columns=self.X.columns)

        if self.weight is not None:
            # If weights are available, apply them.
            X_new = X_new.mul(self.weights_array, axis=1)


        # radius_neighbors returns a list of arrays.
        all_indices = self.nn_radius.radius_neighbors(
            X_new, return_distance=False
        )

        for q_idx, idx in zip(X_new.index, all_indices):
            # If no neighbors are found within the radius, skip or use a fallback.
            if len(idx) == 0:
                # print(f"No neighbors found for query index {q_idx} within radius {self.radius}. Falling back to k-NN.")
                continue  
            neighbors = self.full_df.iloc[idx].copy()
            neighbors['query_index'] = q_idx
            dfs.append(neighbors)

        if not dfs:
            return pd.DataFrame(columns=self.full_df.columns.tolist() + ['query_index'])

        d_test = self.nn_radius.radius_neighbors(
            X_new, return_distance=True
        )[0][0]
        print("test–train distance:", np.percentile(d_test, [10,50,90]))
        return pd.concat(dfs, ignore_index=True)


class MyDatasetNeighbors_kNN:
    """
    A DIY class for neighborhood-based retrieval using radius or k-NN,
    with automatic radius determination.
    """
    def __init__(
        self,
        X: pd.DataFrame,
        weight: pd.DataFrame = None,
        additional_info: Optional[pd.DataFrame | pd.Series] = None,
        k: Optional[int] = None,
        percentage: float = 0.01

    ):
        """
        Initialize with features, predicted labels, and optional settings.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for training samples.
        weight : pd.DataFrame, optional
            Weight matrix for training samples, if applicable.
        additional_info : DataFrame or Series, optional
            Extra columns to keep with the dataset (e.g. metadata).
        k : int or None, default=None
            Number of nearest neighbors to consider. If None, will be set to 5.
        percentage : float in (0,1), default=0.01
            Percentage of the dataset to use for determining the k-th neighbor distance.
        """
        self.X = X
        self.weight = weight
        self.additional_info = additional_info
        self.k = k
        self.scaler = None
        self.percentage = percentage


        # build full dataset for lookup
        self.full_df = X.copy()
        if additional_info is not None:
            self.full_df = pd.concat([self.full_df, additional_info], axis=1)

    def process_X(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_temp = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(X_temp, columns=self.X.columns, index=self.full_df.index)
        if self.weight is not None:
            self.weights_array = np.array([self.weight.get(f, 1.0)
                           for f in self.X.columns], dtype=float)
            self.weights_array = np.sqrt(self.weights_array)
            self.X = self.X.mul(self.weights_array, axis=1)

    def fit(self):
        """
        Determine radius (if needed) and fit the NearestNeighbors model.
        Also plots the k-th neighbor distance distribution with percentile cutoff.
        """
        self.process_X()

        # 1) estimate radius if not provided
        if self.k is None:
            self.k = int(len(self.X) * self.percentage)
            if self.k < 1:
                self.k = 1

            


        # 2) fit k-NN
        self.nn_k = NearestNeighbors(
            n_neighbors=self.k, metric='euclidean'
        ).fit(self.X)

        return self



    def get_neighbors(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve the original training samples that fall in each neighborhood.

        Returns
        -------
        pd.DataFrame
            Concatenated neighbors for all X_new, with an extra column
            'query_index' to indicate which test sample they belong to.
        """
        dfs = []
        if self.scaler is not None:
            # If a scaler is available, standardize the new data first.
            X_new = self.scaler.transform(X_new)
            X_new = pd.DataFrame(X_new, columns=self.X.columns)

        if self.weight is not None:
            # If weights are available, apply them.
            X_new = X_new.mul(self.weights_array, axis=1)


        # radius_neighbors returns a list of arrays.
        all_indices = self.nn_k.kneighbors(
            X_new, return_distance=False
        )

        for q_idx, idx in zip(X_new.index, all_indices):
            # If no neighbors are found within the radius, skip or use a fallback.
            if len(idx) == 0:
                # print(f"No neighbors found for query index {q_idx} within radius {self.radius}. Falling back to k-NN.")
                continue  
            neighbors = self.full_df.iloc[idx].copy()
            neighbors['query_index'] = q_idx
            dfs.append(neighbors)

        if not dfs:
            return pd.DataFrame(columns=self.full_df.columns.tolist() + ['query_index'])
        

        return pd.concat(dfs, ignore_index=True)
