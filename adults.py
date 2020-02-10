#import libraries
import pandas as pd #data Analysis
import numpy as np  #scientific compution
import seaborn as sns #statistical plotting

# IMPORT TRAIN DATA and TAKE A LOOK
adult_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" , names=['age','workclass','fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class'])
adult_data.head()

#replace '?' values to nan
non=[" ?", "?"]
adult_data=adult_data.replace(non, np.nan)  #replace '?' values to nan
adult_data.dropna(inplace=True) #remove instances with missing or malformed features

#Some Features need to be dropped
adult_data=adult_data.drop("native-country" , axis=1)   #deleting a maifold feature
adult_data=adult_data.drop("capital-gain", axis=1)
adult_data=adult_data.drop("capital-loss"  , axis=1)

#Analyzing Data by creating some plots
#For categorized Data
sns.countplot(x="class" , data = adult_data)
#in comparison with Class
sns.countplot(x="class" , hue="sex",  data = adult_data)
sns.countplot(x="class" , hue="race", data=adult_data)
sns.countplot(x="class" , hue="relationship", data=adult_data)
sns.countplot(x="class" , hue="workclass", data=adult_data)
sns.countplot(x="class" , hue="education", data=adult_data)
sns.countplot(x="class" , hue="occupation", data=adult_data)
sns.countplot(x="class" , hue="marital-status", data=adult_data)
sns.countplot(x="class" , hue="native-country", data=adult_data)
#alone
sns.countplot(x="sex",  data = adult_data)
sns.countplot(x="race", data=adult_data)
sns.countplot(x="relationship", data=adult_data)
sns.countplot(x="workclass", data=adult_data)
sns.countplot(x="education", data=adult_data)
sns.countplot(x="occupation", data=adult_data)
sns.countplot(x="marital-status", data=adult_data)
sns.countplot(x="native-country", data=adult_data)      #~99% are from US
#For Continius Data     
adult_data["age"].plot.hist()
adult_data["fnlwgt"].plot.hist(bins=20, figsize=(10,5))
adult_data.info()
adult_data["education-num"].plot.hist()
adult_data["capital-gain"].plot.hist()
adult_data["capital-loss"].plot.hist()
adult_data["hours-per-week"].plot.hist()
adult_data.isin([0]).sum()[10:12] #showing the number of 0 in column 1
(adult_data.sum()[10:12]-adult_data.isin([0]).sum()[10:12])/(adult_data.sum()[10:12])   #as we sww 99.91% of datas in this column are 0

#One-hot encoding for categorical features
sex=pd.get_dummies(adult_data['sex'], drop_first=True)
races=pd.get_dummies(adult_data['race'], drop_first=True)
relation=pd.get_dummies(adult_data['relationship'], drop_first=True)
workclass=pd.get_dummies(adult_data['workclass'], drop_first=True)
education=pd.get_dummies(adult_data['education'], drop_first=True)
occupation=pd.get_dummies(adult_data['occupation'], drop_first=True)
marital=pd.get_dummies(adult_data['marital-status'], drop_first=True)
income=pd.get_dummies(adult_data['class'], drop_first=True)

adult_data=pd.concat([adult_data,sex,races,relation,workclass,education,occupation,marital,income], axis=1 )
adult_data=adult_data.drop(['sex','race','relationship','workclass','education','occupation','marital-status','class'], axis=1)

# Feature_scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
adult_data[['age', 'fnlwgt','education-num','hours-per-week']] = scaler.fit_transform(adult_data[['age', 'fnlwgt','education-num','hours-per-week']])

# IMPORT TEST DATA and TAKE A LOOK
adult_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" , names=['age','workclass','fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class'])
adult_test.head()
adult_test.info()

#replace '?' values to nan
non=[" ?", "?"]
adult_test=adult_test.replace(non, np.nan)  #replace '?' values to nan
adult_test.dropna(inplace=True) #remove instances with missing or malformed features
adult_test['age'] = adult_test['age'].apply(pd.to_numeric) 

#Some Features need to be dropped
adult_test=adult_test.drop("native-country" , axis=1)   #deleting a maifold feature
adult_test=adult_test.drop("capital-gain", axis=1)
adult_test=adult_test.drop("capital-loss"  , axis=1)

#Analyzing Data by creating some plots
#For categorized Data
sns.countplot(x="class" , data = adult_test)
#in comparison with Class
sns.countplot(x="class" , hue="sex",  data = adult_test)
sns.countplot(x="class" , hue="race", data=adult_test)
sns.countplot(x="class" , hue="relationship", data=adult_test)
sns.countplot(x="class" , hue="workclass", data=adult_test)
sns.countplot(x="class" , hue="education", data=adult_test)
sns.countplot(x="class" , hue="occupation", data=adult_test)
sns.countplot(x="class" , hue="marital-status", data=adult_test)
sns.countplot(x="class" , hue="native-country", data=adult_test)
#alone
sns.countplot(x="sex",  data = adult_test)
sns.countplot(x="race", data=adult_test)
sns.countplot(x="relationship", data=adult_test)
sns.countplot(x="workclass", data=adult_test)
sns.countplot(x="education", data=adult_test)
sns.countplot(x="occupation", data=adult_test)
sns.countplot(x="marital-status", data=adult_test)
sns.countplot(x="native-country", data=adult_test)      #~99% are from US
#For Continius Data     
adult_test["age"].plot.hist()
adult_test["fnlwgt"].plot.hist(bins=20, figsize=(10,5))
adult_test.info()
adult_test["education-num"].plot.hist()
adult_test["capital-gain"].plot.hist()
adult_test["capital-loss"].plot.hist()
adult_test["hours-per-week"].plot.hist()
adult_test.isin([0]).sum()[10:12] #showing the number of 0 in column 1
(adult_test.sum()[10:12]-adult_test.isin([0]).sum()[10:12])/(adult_test.sum()[10:12])   #as we sww 99.91% of datas in this column are 0

#One-hot encoding for categorical features
sex=pd.get_dummies(adult_test['sex'], drop_first=True)
races=pd.get_dummies(adult_test['race'], drop_first=True)
relation=pd.get_dummies(adult_test['relationship'], drop_first=True)
workclass=pd.get_dummies(adult_test['workclass'], drop_first=True)
education=pd.get_dummies(adult_test['education'], drop_first=True)
occupation=pd.get_dummies(adult_test['occupation'], drop_first=True)
marital=pd.get_dummies(adult_test['marital-status'], drop_first=True)
income=pd.get_dummies(adult_test['class'], drop_first=True)

adult_test=pd.concat([adult_test,sex,races,relation,workclass,education,occupation,marital,income], axis=1 )
adult_test=adult_test.drop(['sex','race','relationship','workclass','education','occupation','marital-status','class'], axis=1)

# Feature_scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
adult_test[['age', 'fnlwgt','education-num','hours-per-week']] = scaler.fit_transform(adult_test[['age', 'fnlwgt','education-num','hours-per-week']])

#X, y for train data
X_train=adult_data.drop(' >50K', axis=1)
y_train=adult_data[' >50K']

#X, y for test data
X_test=adult_test.drop(' >50K.', axis=1)
y_test=adult_test[' >50K.']

#Implementing logistic regression
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
        
        print('Number of iterations:', n_iterations)
        
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

        print('Correct classifications:', correct)
        print('Incorrect classifications:', incorrect)
        print('Accuracy:', correct/(correct + incorrect))

LR = Logistic_Regression()
w = LR.fit(X_train, y_train)
yh = LR.predict(X_test, w)
LR.evaluate_acc(yh, y_test)

#Implementing Naive Bayes
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
mu_LR, std_LR, mu_NB, std_NB = k_fold_cross_validation(adult_data, 5)
