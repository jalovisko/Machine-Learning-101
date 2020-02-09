
# 1. Acquiring, preprocessing, and analyzing the data
# Importing libraries that are necessary for the project:
import pandas as pd #data Analysis
import numpy as np  #scientific compution
import seaborn as sns #statistical plotting
import matplotlib.pyplot as plt

# IMPORT TRAIN DATA and TAKE A LOOK
iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" , names=["sepal length", "sepal width", "petal length", "petal width", "class"])
iris_data.head()

# Changing dataset to binary classification
for i in range(len(iris_data['class'])):
    y_i = iris_data['class'][i]
    if (y_i == 'Iris-setosa' or y_i == 'Iris-versicolor'):
        iris_data['class'][i] = 0
    elif (y_i == 'Iris-virginica'):
        iris_data['class'][i] = 1
   
# First, we look at the datatype of each column:
iris_data.info()

# The last column has the 'object' type.
iris_data = iris_data.astype({'class': 'int64'})
iris_data.head()

# Splitting the data into the test and train datasets:
msk = np.random.rand(len(iris_data)) < 0.8
train = iris_data[msk]
test = iris_data[~msk]
print('Train entries: ' + str(len(train)))
print('Test entries: ' + str(len(test)))

# Checking how much of the train data are 1 and 0:
X_train = train.iloc[:, :-1] # feature values
y_train = train.iloc[:, -1] # target values
print('Number of Iris-virginica entries: ' + str(train.loc[y_train == 1].shape[0]))
print('Number of other entries: ' + str(train.loc[y_train == 0].shape[0]))

# We can take a look at the prior probability of 'Iris-virginica' entries:
prior = train.loc[y_train == 1].shape[0] / (train.loc[y_train == 1].shape[0] + train.loc[y_train == 0].shape[0])
prior

# We can plot some data to see potential dependencies. In this section dependencies between Pulses i.1 and i.2 are plotted for $i = 2, 3, ..., 17$ ($i=1$ is obscluded since Pulse 1.2 is constant and dropped).
#for i in range(1, train.shape[1] - 1):
#    xlabel = "a"
#    ylabel = "b"
#    fig, ax = plt.subplots()
#    ax2 = train.plot.scatter(x=xlabel, y=ylabel, c = 'class', colormap='viridis', ax = ax)

# 2. Implementing the models
# 2.1. Implementing logistic regression
class Logistic_Regression:
    def __init__(self,
                 lr = 0.01, #learning rate of the gradient descent method
                 eps = 1e-2 # termination condition of the gradient descent method
                ):
        self.lr = lr
        self.eps = eps
        
    # Implementing the logistic function
    def __logistic_function(self, x):
        return 1/(1 + np.exp(-x))
    
    # Implementing the cost function
    def __cost(self,
               w, # N
               X_train, # N x D
               y_train, # N
              ):
        z = np.dot(X_train, w) # N x 1
        J = np.mean(y_train * np.log1p(np.exp(-z)) + (1 - y_train) * np.log1p(np.exp(z)))
        return J
    
    # Implementing the gradient function
    def __gradient(self,
                   w, # N
                   X_train, # N x D
                   y_train, # N
                  ):
        N = y_train.size
        z = np.dot(X_train, w) # N x 1
        yh = self.__logistic_function(z)
        return np.dot(X_train.T, (yh - y_train)) / N
    
    def __GradientDescent(self,
                          X_train, # N x D
                          y_train, # N
                          lr, #learning rate
                          #n_iterations = 300
                          eps# termination condition
                         ):
        #N, D = X.shape
        N = X_train.shape[0]
        
        intercept = np.ones((N, 1))
        X_train = np.concatenate((intercept, X_train), axis=1)
        
        D = X_train.shape[1]
        
        w = np.zeros(D)
        g = np.inf
        
        n_iterations = 0 # just for counting the number of iterations
        
        while (np.linalg.norm(g) > eps):
            g = self.__gradient(w, X_train, y_train)
            w = w - lr * g
            
            n_iterations += 1
        
        print('Number of iterations of the gradient descent:', n_iterations)
        
        return w
    
    # Implementing the fit function
    def fit(self,
            X_train, # N x D
            y_train # N
            ):
        
        w = self.__GradientDescent(X_train, y_train, self.lr, self.eps)
        
        return w
    
    # Implementing the predict function
    def predict (self,
                 X_test,
                 w,
                threshold = 0.5
                ):
        intercept = np.ones((X_test.shape[0], 1))
        X_test = np.concatenate((intercept, X_test), axis=1)

        return self.__logistic_function(np.dot(X_test, w)) >= threshold
    
    # Function for checking the algorithm accuracy
    def evaluate_acc(self,
                    predicted_values,
                    true_values
                    ):
        check = []
        i = 0
        correct = 0
        incorrect = 0
        for y_i in true_values:
            if ((y_i == 1) and (predicted_values[i] == True)):
                check.append([True])
                correct += 1
            elif ((y_i == 0) and (predicted_values[i] == False)):
                check.append([True])
                correct += 1
            else:
                check.append([False])
                incorrect += 1  
            i += 1
        
        accuracy = correct/(correct + incorrect)
        print('Correct classifications:', correct)
        print('Incorrect classifications:', incorrect)
        print('Accuracy of the logistic regression:', correct/(correct + incorrect))
        
        return accuracy    

# Setting up the logistic regression:
LR = Logistic_Regression()

# Running the fit function:
%time w = LR.fit(X_train, y_train)
w

# Separating the feature and the target values in the test dataset:
X_test = test.iloc[:, :-1] # feature values
y_test = test.iloc[:, -1] # target values

# Running the predict function:
pred = LR.predict(X_test, w)

# Checking how much of the test dataset was predicted correctly:
accuracy = LR.evaluate_acc(pred, y_test)

# 2.2. Implementing Naïve Bayes
class GaussianNaiveBayes:
    def __init__(self):
        pass
    
    # The fit function returns the mean, the standard deviation, and the logariphmic prior probability
    def fit(self,
            X_train, # N x D
            y_train, # N x C
           ):
        N = y_train.shape[0]
        c = 0
        C = 1
        D = X_train.shape[1]
        self.mu, self.s = np.zeros((C, D)), np.zeros((C, D)) #mean and standard deviation

        inds = np.nonzero(y_train[:])[0] # indeces of non-zero classes
        self.mu = np.mean(X_train.iloc[inds, :])
        self.s = np.std(X_train.iloc[inds, :])
        self.log_prior = np.log(np.mean(y_train))
        
        return [self.mu, self.s, self.log_prior]
    
    def predict(self,
                X_test, # N_test x
               ):
        log_likelihood = - np.sum( np.log(self.s) + .5*(((X_test - self.mu)/self.s)**2), 1)
        
        pred = self.log_prior + log_likelihood #N_test x C
        
        # Converting to boolean...
        pred = (pred > 0)
        # ... and to binary:
        pred = pred.astype(int)
        return pred #N_test x C
    
    # Function for checking the algorithm accuracy
    def evaluate_acc(self,
                    predicted_values,
                    true_values
                    ):
        check = []
        i = 0 # a counter for retreiving information from predicted target values
        correct = 0
        incorrect = 0
        for y_i in true_values:
            if ((y_i == 1) and (predicted_values.iloc[i] == 1)):
                check.append([True])
                correct += 1
            elif ((y_i == 0) and (predicted_values.iloc[i] == 0)):
                check.append([True])
                correct += 1
            else:
                check.append([False])
                incorrect += 1  
            i += 1

        accuracy = correct/(correct + incorrect)
        print('Correct classifications:', correct)
        print('Incorrect classifications:', incorrect)
        print('Accuracy of the naive Bayes:', correct/(correct + incorrect))
        
        return accuracy

GNB = GaussianNaiveBayes()
mu, s, log_prior = GNB.fit(X_train, y_train)
pred = GNB.predict(X_test)
pred
accuracy = GNB.evaluate_acc(pred, y_test)

# 2.3. Implementing k-fold cross-validation
def k_fold_cross_validation(dataset, # the initial dataset
                           k # number of folds
                           ):
    logistic_regression_accuracy = []
    naive_bayes_accuracy = []
    
    # Splitting the dataset into folds: 
    folds = np.array_split(dataset, k)
    
    for i in range(k):
        print('Fold', i + 1)
        test = folds[i] # the test dataset is the i-th fold
        
        train = folds.copy() # consider the train dataset to be a copy of all folds
        del train[i] # removing the i-th fold
        train = pd.concat(train, sort = False)
        
        # Train dataset:
        X_train = train.iloc[:, :-1] # feature values
        y_train = train.iloc[:, -1] # target values
        
        # Test dataset:
        X_test = test.iloc[:, :-1] # feature values
        y_test = test.iloc[:, -1] # target values
        
        # Logistic regression
        print('Logistic regression:')
        LR = Logistic_Regression()
        w = LR.fit(X_train, y_train)
        pred = LR.predict(X_test, w)
        logistic_regression_accuracy.append(LR.evaluate_acc(pred, y_test))
        
        # Naive Bayes
        print('Naive Bayes')
        GNB = GaussianNaiveBayes()
        mu, s, log_prior = GNB.fit(X_train, y_train)
        pred = GNB.predict(X_test)
        naive_bayes_accuracy.append(GNB.evaluate_acc(pred, y_test))
        print('----------------------------------------')
        
    mu_LR = np.mean(logistic_regression_accuracy)
    std_LR = np.std(logistic_regression_accuracy)
    
    mu_NB = np.mean(naive_bayes_accuracy)
    std_NB = np.std(naive_bayes_accuracy)
    
    print('Logistic regression mean:', mu_LR)
    print('Logistic regression standard deviation:', std_LR)
    
    print('Naive Bayes mean:', mu_NB)
    print('Naive Bayes standard deviation:', std_NB)
    
    return mu_LR, std_LR, mu_NB, std_NB

# 3. Running the experiments
#First we run k-fold cross-validation with k=5:
%time mu_LR, std_LR, mu_NB, std_NB = k_fold_cross_validation(iris_data, 5)

    

