CS 403 - Assignment 1

Shaan Vaidya 150050004

Feature Matrix:
I first tried using the feature matrix as it is, (with the padded ones, of course).
On adding the squares of the columns into the feature matrix, a significantly better result was obtained.
Other functions - polynomial terms, exponential, cosh, etc. were tried without any significant improvement to the solution

L2 gradient descent and L2 closed form:
0.00368823 is the norm of the difference of the two W's. The Gradient Descent almost converges to the value obtained by the closed form solution. Thus, they are almost equal as expected (Smaller Lambda yields smaller value for this norm). 

L2 gradient descent and Lp gradient descent:
Lp (p<2) performed better than L2 GD on Kaggle, best solution came for p = 1.7

Fine tuning lambda:
Different values of lambda were tried. Too high a value penalises the parameters too much, while a very small lambda may lead to overfitting and higher MSE for test data, though it works better on train data. 
For the learning rate, a constant value was used. A line-search expression for the same would have given a quicker solution (steepest descent) but the no significant change to the final result. 
K-fold cross validation technique could have been implemented here, by treating 1/kth of the sample as validation data each of the k times. 

Other observations:
One interesting observation or rather anomaly, I made, was regarding the 'nan' output error with the gradient descent algorithm. Some values working on other computers weren't working on mine. In other words, it seems that the fact that the learning rate is too high depends in some way on the machine too.