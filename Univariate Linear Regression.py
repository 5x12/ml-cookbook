'''
This is a template for Univariate Linear Regression that calculates 
Cost fuction J(theta) and gradient descent
 
'''

import numpy as np
import pandas as pd



def cost_function(X, Y, theta):
    
    predictions = X.dot(theta)   # [X] * [theta]
    sqrError = np.square(predictions-Y)
    J = (1/(2*m))*sum(sqrError)
    
    return J

                   
def gradient_descent(X, Y, theta, alpha, num_iters):
    
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)  # [X] * [theta] , not [X] * . [theta]

        errors_x1 = np.multiply((predictions - Y), X[:, 0])
        errors_x2 = np.multiply((predictions - Y), X[:, 1])

        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * sum(errors_x1)
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * sum(errors_x2)

        J_history[i, 0] = cost_function(X, Y, theta)

    return theta, J_history


#uploading data

data = pd.read_csv('data1.csv')
dataMatrix = np.matrix(data) 


#defining variables: X, Y, theta

X = dataMatrix[:,0]
Y = dataMatrix[:,1]
m = X.size
x0 = np.ones([96,1])
X=np.concatenate((x0, X), axis = 1)

theta = np.zeros(shape=(2,1))


'''
Cost Function

'''

J = cost_function(X, Y, theta)
print('This is our Cost Function:', J)


'''
Gradient Descent

'''  

#Some gradient descent settings and calculation
num_iters = 1500
alpha = 0.01


theta, J_history = gradient_descent(X, Y, theta, alpha, num_iters)

print(J_history)
print(theta)

