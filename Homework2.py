# Dr Jianhua Chen
# Programming assignment part a
# Student name: Seyedeh Shaghayegh Rabbanian 899645944 (srabba2@lsu.edu)
# CSC 7333


# Importing packages which are needed for programming
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Part a (Reading the dataset KCSmall2.csv)
mydataset= pd.read_csv('KCSmall2.csv',header=None)
x = mydataset.iloc[:,0]
y = mydataset.iloc[:,1]



# General code :Part a (Performing  gradient descent to learn parameter vector (theta0 = b ,theta1 = a)
# First we initialize values for a and be to be 0. We specify values for number of iterations and learning rate and m which is the number of 
# training examples equals to length of x. After that, by using for loop we calculate our predicted value for y, loss (error) function and our 
# derivatives. Then, by using update rule, we update the values for our parameters. This for loop continues until we complete our number of iterations
#def gradient_descent(x,y):
#    b_curr = a_curr = 0
#    iterations = 15
#    m = len(x)
#    learning_rate = 0.1
#
#    for i in range(iterations):
#        y_predicted = a_curr * x + b_curr
#        loss = (1/(2*m)) * sum([val**2 for val in (y-y_predicted)])
#        ad = -(1/m)*sum(x*(y-y_predicted))
#        bd = -(1/m)*sum(y-y_predicted)
#        a_curr = a_curr - learning_rate * ad
#        b_curr = b_curr - learning_rate * bd
#        print ("a {},b {}, loss {}, iteration {}".format(a_curr,b_curr,loss,i))

#gradient_descent(x,y)

# Desired output of the program
# (1) Plot the data

print('(1) Plot the data')
plt.plot(x,y,'X')
plt.title('House Price Prediction from Living Area')
plt.xlabel('House Living Area in 1000 Square Feet')
plt.ylabel('House Prices in 10,000 Dollars')
plt.show()

print('........................................................................................................')

# (2) Print the loss function for (0,0) and (-1,20)

print('(2) Print the loss function for (0,0) and (-1,20)')
m = len(x)
y_predicted2 = (0) * x + 0
loss2 = (1/(2*m)) * sum([val**2 for val in (y-y_predicted2)])
print(loss2)

y_predicted3 = (20) * x -1
loss3 = (1/(2*m)) * sum([val**2 for val in (y-y_predicted3)])
print(loss3)

print('........................................................................................................')

# (3) Plot the loss function for n=15 iterations (alpha= 0.01,0.1,0.2,0.4) and print out parameters (I will get learning rate by using input but I know that learning rate=0.2 has the least cost, because I already run the code multiple times and screenshots are in Word document.)

print('(3) Plot the loss function for n=15 iterations (alpha= 0.01,0.1,0.2,0.4) and print out parameters (I will get learning rate by using input but I know that learning rate=0.2 has the least cost, because I already run the code multiple times and screenshots are in Word document.)')
def gradient_descent(x,y):
    b_curr = a_curr = 0
    iterations = 15
    m = len(x)
    learning_rate = float(input('Enter learning rate value: '))
    loss_list = []


    for i in range(iterations):
        y_predicted = a_curr * x + b_curr
        loss = (1/(2*m)) * sum([val**2 for val in (y-y_predicted)])
        loss_list.append(loss)
        ad = -(1/m)*sum(x*(y-y_predicted))
        bd = -(1/m)*sum(y-y_predicted)
        a_curr = a_curr - learning_rate * ad
        b_curr = b_curr - learning_rate * bd
        print ("a {},b {}, loss {}, iteration {}".format(a_curr,b_curr,loss,i))

    plt.plot(list(range(iterations)),loss_list,'X')
    plt.title('Loss Function Curve for 15 iterations (alpha = '+ str(learning_rate) + ')')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.show()
    



    print('........................................................................................................')

# (4) Predict y_predicted for x=3.5 and x=7
    print('(4) Predict y_predicted for x=3.5 and x=7')
    New_x=[3.5,7]
    for l in New_x: 
        price=a_curr*l+b_curr
        print("Living area: {}, Price: {}".format(l,price))
      

gradient_descent(x,y)












