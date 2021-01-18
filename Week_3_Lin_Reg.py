# Marks Predictor

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Load Dataset

path = r"https://drive.google.com/uc?export=download&id=13ZTYmL3E8S0nz-UKl4aaTZJaI3DVBGHM"

df  = pd.read_csv(path)


df.head()


df.tail()


df.shape


# ## Discover and visualize the data for insights

df.info()


df.describe()


plt.scatter(x =df.study_hours, y = df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()


# ## Prepare the data for Machine Learning algorithms 
# Data Cleaning
df.isnull().sum()



df.mean()



df2 = df.fillna(df.mean())



df2.isnull().sum()


# In[14]:


df2.head()


# split dataset


X = df2.drop("student_marks", axis = "columns")
X

y = df2.drop("study_hours", axis = "columns")
y

print("shape of X = ", X.shape)
print("shape of y = ", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=51)

print("shape of X_train = ", X_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of X_test = ", X_test.shape)
print("shape of y_test = ", y_test.shape)


# # Select a model and train it

# y = m * x + c
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)

lr.coef_

lr.intercept_

lr.predict([[4]])[0][0].round(2)

y_pred  = lr.predict(X_test)
y_pred

pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])


lr.score(X_test,y_test)

plt.scatter(X_train,y_train)


plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = "r")


# ## Save Ml Model


import joblib
joblib.dump(lr, "student_mark_predictor.pkl")

model = joblib.load("student_mark_predictor.pkl")


model.predict([[6]])[0][0]




# *** Mult Lin Reg *** #




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import os
print(os.getcwd())

os.chdir('F:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')

print(os.getcwd())

import math
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.set_style('whitegrid')



# Importing the dataset
dataset = pd.read_csv('VC_Startups.csv')

dataset

r_d = dataset.iloc[:, 0]
r_d

x = r_d.values 
x

type(x)

prft = dataset.iloc[:, 4]
prft

y = prft.values
y

import matplotlib.pyplot as plt
plt.scatter(x,y,label='',color='k',s=100)
plt.xlabel('R&D')
plt.ylabel('Profit')
plt.title('Profit vs R&D Spend')
plt.legend()
plt.show()


admn = dataset.iloc[:, 1]
admn

x = admn.values
x
                
plt.scatter(x,y,label='',color='k')
plt.xlabel('Admin')
plt.ylabel('Profit')
plt.title('Profit vs Admin Spend')
plt.legend()
plt.show()


mrktng = dataset.iloc[:, 2]
x = mrktng.values
plt.scatter(x,y,label='',color='k')
plt.xlabel('Marketing')
plt.ylabel('Profit')
plt.title('Profit vs Marketing Spend')
plt.legend()
plt.show()


x = dataset.iloc[:, 3].values
plt.scatter(x,y,label='',color='k')
plt.xlabel('State')
plt.ylabel('Profit')
plt.title('Profit vs State')
plt.legend()
plt.show()



df = dataset.iloc[:, 3:5]
df.boxplot(column='Profit',by='State')


'''half_masked_corr_heatmap(dataset,
                         'VC Data - Variable Correlations',
                         )
#plot_correlation(dataset,'Profit')'''

#***



# Scikit Learn again

X = dataset.iloc[:, :-1].values
X

y = dataset.iloc[:, 4].values
y

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

X_train

X.shape

X = dataset.iloc[:, :-2].values
X

X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

print(regressor.coef_)

print(regressor.intercept_)


#import statsmodels.formula.api as sm
#from statsmodel.api import OLS
#sm.OLS(y,X).fit().summary()

#***



# Dummy Vars & Encoders
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dataset2 = ['Pizza','Burger','Bread','Bread','Bread','Burger','Pizza','Burger']
dataset2

values = array(dataset2)
print(values)

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)

onehot = OneHotEncoder(sparse=False)

onehot_encoded = onehot.fit_transform(integer_encoded)
print(onehot_encoded)

#inverted_result = label_encoder.inverse_transform([argmax(onehot_encoded[0,:])])
#print(inverted_result)

X = dataset.iloc[:, :-1].values
X

y = dataset.iloc[:, 4].values
y

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

X

X[:, 3] = labelencoder.fit_transform(X[:, 3])

X

'''onehotencoder = OneHotEncoder(categorical_features = [3])
tmpDF = pd.DataFrame(X)
tmpDF 
X = onehotencoder.fit_transform(X).toarray()'''

X
print('--- OneHotEncoder ---')
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('State', OneHotEncoder(), [3])], remainder='passthrough')

X = ct.fit_transform(X) #.toarray()

print(X)


tmpDF = pd.DataFrame(X)
tmpDF

X.shape


# Avoiding the Dummy Variable Trap
X = X[:, 1:]

X.shape

X

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

print(regressor.coef_)

