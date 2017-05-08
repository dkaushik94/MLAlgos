
"""Regression using Linear Gradient Descent using two variables."""

# Author: Debojit Kaushik(8th May 2017)

import numpy as np
import matplotlib.pyplot as plt


'''
    Cost function calculator. Parameters : slope and constant of the regression line 
    ie. the parameters for the hypotheses.
    min(J(x)) = 1/(m*2) (Summation from i =0 to class size) d/d(theta)(h(x)-y).
'''
def costFunction(theta0, theta1, mode, x, y, setSize):
    temp = np.float64(0)
    if mode and mode is True:
        try:
            for it, item in enumerate(x):
                temp += theta0 + theta1*x[it] - y[it]
            temp = temp/setSize
            return temp
        except Exception as e:
            print(e)
            raise Exception("Cost function failed.")
    elif not mode and mode is False:
        try:
            for it, item in enumerate(x):
                temp += (theta0 + theta1*x[it] - y[it])*x[it]
            temp = temp/setSize
            return temp
        except Exception:
            print(e)
            raise Exception("Cost function failed.")
    else:
        raise ValueError("No Valid 'mode' parameter select.Eg: mode = True or False")



'''
    Convergene condition:
    If current slop - previous slope is < 0.01, then state convergence is true, and stop iterations.
'''
def convergence(prevTheta1, currTheta1):
    '''
        Method to check for convergence condition. 
    '''
    prevTheta1 = abs(prevTheta1)
    currTheta1 = abs(currTheta1)
    conv = max(prevTheta1,currTheta1) - min(prevTheta1, currTheta1)

    if conv<0.0001:
        return True
    else:
        return False


def regressionLine(slope, intercept, plot):
    x = np.arange(50)
    y = []
    for point in range(50):
        temp = slope*x[point] + intercept
        y.append(temp)
    plot.plot(x,y)



'''
    Gradient descent for local minima.
    Iterate and compute theta0, theta1 until convergence condition is satisfied.
'''
def gradientDescent(price, area, plot):
    try:
        regressionParameters = []
        theta0 = np.float64(0)
        theta1 = np.float64(0)
        m = len(price)
        alpha = 0.001
        temp0 = np.float64(0)
        temp1 = np.float64(0)
        J0 = np.float64(0)
        J1 = np.float64(0)
        converge = False
        while converge is False:
            J0 = costFunction(theta0, theta1, True, price, area, m)
            J1 = costFunction(theta0, theta1, False, price, area, m)
            temp0 = theta0 - alpha*J0
            temp1 = theta0 - alpha*J1
            converge = convergence(theta1, temp1)
            theta0 = temp0
            theta1 = temp1
            regressionLine(theta1, theta0, plot)
        return theta0,theta1
    except Exception as e:
        print(e)
        raise Exception("Please check parameters.")
        




if __name__ == '__main__':
    print("Linear Regression Prediction using Linear Gradient Descent\n")
    try:
        dataSet = np.genfromtxt('/home/debojit/ML/MLAlgos/kc_house_data.csv', delimiter = ',', names = True, dtype = 'float64')
        price, area = np.zeros([len(dataSet),1]),np.zeros([len(dataSet),1])        
        for it, l in enumerate(dataSet):
            price[it] = l[2]
            area[it] = l[5]
        
        a = np.random.rand(50)*40
        b = np.random.rand(50)*40
        plt.plot(a,b,'o')
        theta0, theta1 = gradientDescent(a,b, plt)
        print("Theta0: %s Theta1: %s" %(theta0, theta1))
        # x,y = regressionLine(theta1,theta0)
        # plt.plot(x,y,'-')
        plt.show()
    except Exception:
        raise Exception("Something is wrong.")
