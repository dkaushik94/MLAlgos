
"""Regression using Linear Gradient Descent using two variables."""

# Author: Debojit Kaushik(8th May 2017)

import numpy as np
import matplotlib.pyplot as plt
import math



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
                temp += (theta0 + theta1*x[it] - y[it])
            return temp
        except Exception as e:
            print(e)
            raise Exception("Cost function failed.")
    elif not mode and mode is False:
        try:
            for it, item in enumerate(x):
                temp += (theta0 + theta1*x[it] - y[it])*x[it]
            return temp
        except Exception:
            print(e)
            raise Exception("Cost function failed.")
    else:
        raise ValueError("No Valid 'mode' parameter select.Eg: mode = True or False")



'''
    Convergence condition:
    If current slop - previous slope is < 0.000001, then state convergence is true, and stop iterations.
'''
def convergence(prevTheta1, currTheta1):
    '''
        Method to check for convergence condition. 
    '''
    prevTheta1 = abs(prevTheta1)
    currTheta1 = abs(currTheta1)
    conv = max(prevTheta1,currTheta1) - min(prevTheta1, currTheta1)

    if conv<0.000001:
        return True
    else:
        return False

'''
    Line plotting before and after.
'''
def regressionLine(slope, intercept, plot, i, color=None):
    x = np.arange(i)
    y = []
    for point in range(i):
        temp = slope*x[point] + intercept
        y.append(temp)
    if color is None:
        plot.plot(x,y)
    else:
        plot.plot(x,y,color=color)



'''
    Error function for hypotheses.
'''
def errorValue(x, y, theta0, theta1, setSize, it, plot):
    Error = 0
    try:
        for i in range(setSize):
            Error += (y[i] - (theta1*x[i] + theta0)) ** 2
        err = Error / setSize
        plot.plot(err, it, 'go')
    except Exception as e:
        print(e)




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
        alpha = 0.0001
        temp0 = np.float64(0)
        temp1 = np.float64(0)
        J0 = np.float64(0)
        J1 = np.float64(0)
        converge = False
        error = np.empty([])
        i = 0
        while converge is False:
            J0 = costFunction(theta0, theta1, True, price, area, m)
            J1 = costFunction(theta0, theta1, False, price, area, m)
            print("Iteration number: %s" %i)
            
            #Calculate error from cost function. Should be decreasing WRT to iterations.
            errorValue(price, area, theta0, theta1, m, i, plot)
            
            temp0 = theta0 - (alpha*J0/m)
            temp1 = theta0 - (alpha*J1/m)
            
            #convergence test. If slop is changing very minutely then stop.
            converge = convergence(theta1, temp1)
            
            theta0 = temp0
            theta1 = temp1
            i += 1

            if math.isnan(theta0) or math.isnan(theta1) is True:
                break
            else:
                pass
        return theta0,theta1, i
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
        
        
        ''' Feature Scaling. 
            (
                Every entry divded my highest number of the set. 
                Another option is to subtract mean of set from eveery element and divide it by range of values for that column.
            )
        '''
        price = price/max(price)
        area = area/max(area)
        
        theta0, theta1, i = gradientDescent(price, area, plt)
        print("Theta0: %s Theta1: %s" %(theta0, theta1))
        regressionLine(theta1,theta0, plt, i, color = 'red')
        plt.show()
    except Exception:
        raise Exception("Something is wrong.")
