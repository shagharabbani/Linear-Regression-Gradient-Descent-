# Dr Jianhua Chen
# Programming assignment part b
# Student name: Seyedeh Shaghayegh Rabbanian 899645944 (srabba2@lsu.edu)
# CSC 7333

# Importing packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

# (1) Loading dataset
print('(1) Print out the first 5 rows of the raw input data ')
mydataset= pd.read_csv('KCSmall_NS2.csv',header=None)
# Print out the first 5 rows of the raw input data
print(mydataset[0:5])

print('..................................................................................................')

x = mydataset.iloc[:,0:3]
y = mydataset.iloc[:,3]

# (2) Standardizing input data
print('(2) Print out the first 5 rows of normalized data (after adding the dummy 0th column)')

x_scaled = preprocessing.scale(x)


# Defining a dummy column ( Column of all 1)
dummy_column = np.ones((100,1),dtype='int32')


# Adding the dummy column to the scaled matrix
Final_x = np.append(dummy_column,x_scaled,axis = 1)
print(Final_x[0:5])

print('..................................................................................................')

# (3) Print out the cost J value for theta = (0,0,0,0)

print('(3) Print out the cost  J value for 𝜃= (0, 0, 0, 0)')

n= Final_x.shape[1]
theta = np.zeros(n)
print(theta)

def cost_function(Final_x,y):
    m=len(y)
    loss=np.sum((Final_x.dot(theta)-y)**2)/(2*m)
    print(loss)

cost_function(Final_x,y)

print('..................................................................................................')

# (4) Run gradient descent for n=50 iterations with alpha=(0.01,0.1,0.5,1,1.5)

print('(4) Run gradient descent for n=50 iterations with alpha=(0.01,0.1,0.5,1,1.5). Learning rate=1 works the best.')

def gradient_descent(x,y):
    theta = np.zeros(n)
    iterations = 50
    m = len(y)
    learning_rate = float(input('Enter learning rate value: '))
    loss_list = []

    for i in range(iterations):
        y_predicted = Final_x.dot(theta)
        loss=np.sum((Final_x.dot(theta)-y)**2)/(2*m)
        loss_list.append(loss)
        thetad = (1/m)*((Final_x.T).dot(y_predicted-y))
        theta = theta - learning_rate * thetad
        print ("Theta{}, loss {}, iteration {}".format(theta,loss,i))

    plt.plot(list(range(iterations)),loss_list,'X')
    plt.title('Loss Function Curve for 50 iterations (alpha = '+ str(learning_rate) + ')')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.show()


gradient_descent(x,y)

print('..................................................................................................')

# (5) Print out the predicted y value for the input n_bed=3, liv_area=2000, lot_area=8550.
# First we should standardize data point.

print('(5) Print out  the predicted y value for the input ')

x_test = np.array([3, 2000, 8550])
colaverage = x.mean(axis = 0)
colstd = x.std(axis = 0)
standard = (x_test-colaverage)/colstd

#Adding dummy column to standardized matix
dummy_column2 = np.array([1])
Final_xtest = np.append(dummy_column2,standard,axis = 0)
print(Final_xtest)

New_theta = np.array([519250.55 ,7719.29675092,166327.3196951,53701.10923516])
print(New_theta)
Predictedvalue = Final_xtest.dot(New_theta.T)
print(Predictedvalue)


    
     

