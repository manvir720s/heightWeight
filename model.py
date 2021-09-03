#https://towardsdatascience.com/simple-linear-regression-in-python-8cf596ac6a7c
#https://intellipaat.com/blog/what-is-linear-regression/
#http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# Importing the dataset
dataset = pd.read_csv('Data_Height_Weight.csv')

dataset = dataset.drop(dataset.columns[0], axis = 1)
dataset.drop_duplicates(inplace=True)

X = pd.DataFrame(dataset['Height(Inches)'])
y = pd.DataFrame(dataset['Weight(Pounds)'])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

def classify(a):
    arr = np.array([a]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = regressor.predict(query)[0] # Retrieve from dictionary
    if (prediction < 0):
        prediction = -prediction
    return prediction # Return the prediction


# Visualising the Training set results
'''
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Height vs Weight (Training set)')
plt.xlabel('Height (inches)')
plt.ylabel('Weight')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Height vs Weight (Test set)')
plt.xlabel('Height (inches)')
plt.ylabel('Weight')
plt.show()

r_sq = regressor.fit(X_train, y_train).score(X_train, y_train) # closer to one is better 
print('coefficient of determination:', r_sq)
'''

