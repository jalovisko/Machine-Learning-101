#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:43:54 2020

@author: Nick

# EXAMPLE

MBM = Naive_Bayes_Mixed(categorical_features = [0])
priors, means, stds = MBM.fit(X_train, y_train)
MBM.predict(X_test, priors, means, stds)

"""
import numpy as np

class Naive_Bayes_Mixed():
    def __init__(self,
                 categorical_features = None,
                 alpha = 0.5,
                 ):
        self.categorical_features = categorical_features
        self.alpha = alpha
    
    def fit(self, X_train, y_train): 

        self.epsilon = 1e-9 * np.var(X_train, ddof = 1, axis = 0).max()
        
        # Get whatever that is needed
        uniques = np.unique(y_train)
        num_classes = uniques.size
        (num_samples, num_features) = X_train.shape

        # Correct the inputs
        priors = np.bincount(y_train)/num_samples

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
        max_categories = np.max(X_train.to_numpy()[:, self.categorical_features],axis=0) + 1
        max_categories = max_categories.astype(int)

        # Prepare empty arrays
        if self.gaussian_features.size != 0:
            means = np.zeros((num_classes, len(self.gaussian_features)))
            stds = np.zeros((num_classes, len(self.gaussian_features)))
        if self.categorical_features.size != 0:
            self.categorical_posteriors = [
                    np.zeros((num_classes, num_categories))
                    for num_categories in max_categories]

        # TODO optimise below!
        for y_i in uniques:

            if self.gaussian_features.size != 0:
                x = X_train.to_numpy()[y_train == y_i, :][:, self.gaussian_features]
                means[y_i, :] = np.mean(x, axis=0)
                # note: it's really sigma squared
                stds[y_i, :] = np.var(x, axis=0)

            if self.categorical_features.size != 0:
                for i, categorical_feature in enumerate(self.categorical_features):
                    dist = np.bincount(
                            X_train.to_numpy()[y_train == y_i, :][:, categorical_feature].astype(int),
                            minlength = max_categories[i]) + self.alpha
                    self.categorical_posteriors[i][y_i, :] = dist / np.sum(dist)

        return priors, means, stds
    
    def __Gaussian_Naive_Mixed(self,
                               X_test,
                               priors,
                               means,
                               stds
                               ):

        X_test = np.array(X_test)

        if self.gaussian_features.size != 0:
            # TODO optimisation: Below is a copy. Can consider masking
            x_gaussian = X_test[:, self.gaussian_features]
            mu = means[:, np.newaxis]
            s = stds[:, np.newaxis] + self.epsilon

            # Likelihood from the test dataset
            likelihood = 1./np.sqrt(2.*np.pi*s) * \
                np.exp(-((x_gaussian-mu)**2.)/(2.*s))

            t = np.prod(likelihood, axis = 2)[:, :, np.newaxis]
            t = np.squeeze(t.T)

        if self.categorical_features.size != 0:

            # Cast tensor to int
            X = X_test[:, self.categorical_features].astype(int)

            # A list of length=num_features.
            # Each item in the list contains the distributions for the y_classes
            # Shape of each item is (num_classes,1,num_samples)
            preds = [categorical_posterior[:, X[:, i][:, np.newaxis]]
                      for i, categorical_posterior
                      in enumerate(self.categorical_posteriors)]

            r = np.concatenate([preds], axis = 0)
            r = np.squeeze(r, axis = -1)
            r = np.moveaxis(r, [0, 1, 2], [2, 0, 1])

            p = np.prod(r, axis = 2).T

        if (self.gaussian_features.size != 0) and (self.categorical_features.size != 0):
            finals = t * p * priors
        elif (self.gaussian_features.size != 0):
            finals = t * priors
        elif (self.categorical_features.size != 0):
            finals = p * priors

        normalized = finals.T/(np.sum(finals, axis = 1) + 1e-6)
        normalized = np.moveaxis(normalized, [0, 1], [1, 0])

        return normalized
    
    def predict(self,
                X_test,
                priors,
                means,
                stds
                ):
        
        pred = self.__Gaussian_Naive_Mixed(X_test,
                                           priors,
                                           means,
                                           stds)
        return np.argmax(pred, axis = 1)