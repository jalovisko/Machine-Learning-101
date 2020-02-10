#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:43:54 2020

@author: Nick
"""
import numpy as np

class Naive_Bayes_Mixed():
    def __init__(self,
                 categorical_features = None,
                 alpha = 0.5,
                 ):
        self.categorical_features = categorical_features
        self.alpha = alpha
    
    def fit(self, X, y): 

        # Get whatever that is needed
        uniques = np.unique(y)
        num_classes = uniques.size
        (num_samples, num_features) = X.shape

        # Correct the inputs
        priors = np.bincount(y)/num_samples

        if self.categorical_features is None:
            self.categorical_features = []
        elif self.categorical_features is 'all':
            self.categorical_features = np.arange(0, num_features)

        # Get the index columns of the discrete data and continuous data
        self.categorical_features = np.array(
                self.categorical_features).astype(int)
        self.gaussian_features = np.delete(
                np.arange(num_features),
                self.categorical_features)

        # How many categories are there in each categorical_feature
        # Add 1 due to zero-indexing
        max_categories = np.max(X.to_numpy()[:, self.categorical_features], axis=0) + 1
        max_categories = max_categories.astype(int)

        # Prepare empty arrays
        if self.gaussian_features.size != 0:
            w = np.zeros((num_classes, len(self.gaussian_features)))
            sigma = np.zeros((num_classes, len(self.gaussian_features)))
        if self.categorical_features.size != 0:
            categorical_posteriors = [
                    np.zeros((num_classes, num_categories))
                    for num_categories in max_categories]

        # TODO optimise below!
        for y_i in uniques:

            if self.gaussian_features.size != 0:
                x = X.to_numpy()[y == y_i, :][:, self.gaussian_features]
                w[y_i, :] = np.mean(x, axis=0)
                # note: it's really sigma squared
                sigma[y_i, :] = np.var(x, axis=0)

            if self.categorical_features.size != 0:
                for i, categorical_feature in enumerate(self.categorical_features):
                    dist = np.bincount(
                            X.to_numpy()[y == y_i, :][:, categorical_feature].astype(int),
                            minlength = max_categories[i]) + self.alpha
                    categorical_posteriors[i][y_i, :] = dist / np.sum(dist)

        return priors, w