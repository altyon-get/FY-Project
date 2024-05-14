import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes =datasets.load_diabetes()

# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

# diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X=diabetes.data
# print(diabetes_X)
diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]


diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-30:]


model = linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predict = model.predict(diabetes_X_test)


x = mean_squared_error(diabetes_Y_test,diabetes_Y_predict)
print("Mean squared error: ",x)


print("Weights:",model.coef_)
print("Intercept:", model.intercept_)

# plt.scatter(diabetes_X_test,diabetes_Y_test)
# plt.savefig('plot.png')

# Mean squared error:  3035.060115291269
# Weights: [941.43097333]
# Intercept: 153.39713623331644

# Mean squared error:  1826.4841712795046
# Weights: [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
#   458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
# Intercept: 153.05824267739402
