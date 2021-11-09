#  librires 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_theme(style="darkgrid")

## Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# other libraires
from scipy.stats import skew, kurtosis

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore')

train = pd.read_csv('Train.csv')
test = pd.read_csv('test.csv')

orginal_train = train.copy()
orginal_test = test.copy()

# understanding our data by small steps and visual tools
def understand_our_data(data):
    print('Our Data Info','\n')
    print(data.info(),'\n')
    print('Describe our Numeric data','\n')
    print(data.describe(),'\n')
    print('Describe our Objectiv data','\n')
    print(data.describe(include=['O']),'\n')
    print('Objects columns','\n')
    print(data.dtypes == 'object','\n')
    print('Sorted type of columns','\n')
    print(data.dtypes.sort_values(),'\n')
    print('Number of null values','\n')
    print(data.isna().sum().sort_values(),'\n')
    print('Shape of our Data','\n')
    print(data.shape,'\n')
    print('Percnt of test data comparing of train data','\n')
    print(test.shape[0]/data.shape[0],'\n')
    print('Number of unique vales','\n')
    print(data.nunique().sort_values(),'\n')
    print('percantge of null values', '\n')
    print(round(data.isna().sum(axis=0)/len(data),2)*100)

understand_our_data(train)
understand_our_data(test)

# visulization our  numaeric data
def plot_hist_boxplot(column):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.distplot(train[train[column].notnull()][column], ax=ax[0])
    sns.boxplot(train[train[column].notnull()][column], ax=ax[1])
    print('skewness for {} : '.format(column),skew(train[train[column].notnull()][column]))
    print('kurtosis for {} : '.format(column), kurtosis(train[train[column].notnull()][column]))
    plt.show()

num_col = ['Outlet_Establishment_Year','Item_Weight', 'Item_Visibility',
           'Item_MRP','Item_Outlet_Sales']
for col in num_col:
    plot_hist_boxplot(col)
    
# edit our category data    
train['Item_Fat_Content'].replace(to_replace=['LF', 'low fat', 'reg'],
                                  value=['Low Fat', 'Low Fat', 'Regular'], inplace=True)
train['Item_Fat_Content'].value_counts()

test['Item_Fat_Content'].replace(to_replace=['LF', 'low fat', 'reg'],
                                  value=['Low Fat', 'Low Fat', 'Regular'], inplace=True)
test['Item_Fat_Content'].value_counts()

train['Item_Type_combined']= train['Item_Identifier'].apply(lambda x : x[0:2])
train['Item_Type_combined'].replace(to_replace=['FD', 'DR', 'NC'],
                                    value=['Food', 'Drinks', 'Non_consumable'], inplace=True)

test['Item_Type_combined']= test['Item_Identifier'].apply(lambda x : x[0:2])
test['Item_Type_combined'].replace(to_replace=['FD', 'DR', 'NC'],
                                    value=['Food', 'Drinks', 'Non_consumable'], inplace=True)

train.drop(columns='Item_Type', axis=1, inplace=True) 
test.drop(columns='Item_Type', axis=1, inplace=True) 

 
# visualize category columns  
def uniqe_columns(column):
    print('count of uniqe values for {}'.format(column), '/n')
    print(train.groupby(column)[column].count(), '/n')
    
obj_col = train.columns.difference(num_col).tolist()
obj_col_count = ['Outlet_Location_Type','Outlet_Type','Outlet_Size',
                 'Item_Fat_Content','Outlet_Identifier','Item_Type_combined']

for col in obj_col_count:
    uniqe_columns(col)
    sns.countplot(col, data=train)
    plt.xticks(rotation=45)
    plt.title(col)
    plt.show()
    
#value of sales increases for the increase in MRP of the item
plt.scatter(train.Item_MRP,train.Item_Outlet_Sales,c='g')
plt.show()

# realtion with our target 
def facegrid(column):
    sns.FacetGrid(train, col=column, size=3,
                  col_wrap=5).map(plt.hist,'Item_Outlet_Sales').add_legend()

col = ['Outlet_Type', 'Item_Fat_Content', 'Outlet_Size','Outlet_Location_Type',
       'Item_Type_combined'] 
for column in col:
    facegrid(column)    

  
# corrlation 
corr = train.corr()
corr = train.corr().unstack().reset_index() # to mkae it column

# correlation
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

# checking outliers 
factor = 3
def outliers(data, column):
    upper_lim = data[column].mean () + data[column].std () * factor
    lower_lim = data[column].mean () - data[column].std () * factor
    data = data[(data[column] > upper_lim) & (data[column] < lower_lim)]
    return data

outliers(data=train, column='Item_Outlet_Sales')
outliers(data=train, column='Item_Visibility')

# handle null values
train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode(), inplace=True)

test['Item_Weight'].fillna(test['Item_Weight'].median(), inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode(), inplace=True)

train_num = train[num_col]
test_num = test[['Outlet_Establishment_Year','Item_Weight', 'Item_Visibility',
           'Item_MRP']]

# label encoder our data test and train
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

for col in obj_col_count:
    label.fit(train[col])
    train[col] = label.transform(train[col])

train_new = train.drop('Item_Identifier', axis=1)
train_new = pd.get_dummies(train_new,columns=['Outlet_Identifier'])


for col in obj_col_count:
    label.fit(test[col])
    test[col] = label.transform(test[col])

test_new = test.drop('Item_Identifier', axis=1)
test_new = pd.get_dummies(test_new,columns=['Outlet_Identifier'])

train_new_1 = train_new.drop('Item_Outlet_Sales', axis=1)
train_new_1['Item_Weight'].fillna(train_new_1['Item_Weight'].median(), inplace=True)
y_train = train_new['Item_Outlet_Sales']

'''
columns have null values ['Item_Weight', 'Outlet_Size'] [17%,28%] in train and test
 we did not visualize it [Item_Identifier]
 need to edit it [Item_Fat_Content]
 we have good correlation between Item_MRP and Item_Outlet_Sales 0.57
 outliers mayber we have in [Item_Visibility,Item_Outlet_Sales]

'''

# Model our data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet 
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import xgboost as xgb

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
#import featuretools as ft
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

models = [('lr',LinearRegression()),('ridge',Ridge()),('rfr',RandomForestRegressor()),
          ('etr',ExtraTreesRegressor()),('br',BaggingRegressor()),
          ('gbr',GradientBoostingRegressor()),('en',ElasticNet()),('mlp',MLPRegressor())]


#Making function for making best 2 models for further hyperparameter tuning
def basic_model_selection(x,y,cross_folds,model):
    scores=[]
    names = []
    for i , j in model:
        cv_scores = cross_val_score(j, x, y, cv=cross_folds,n_jobs=5)
        scores.append(cv_scores)
        names.append(i)
    for k in range(len(scores)):
        print(names[k],scores[k].mean())
        
basic_model_selection(train_new_1,y_train,4,models)

#Average score for XGBoost matrix
# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=train_new_1,label=y_train)
# import XGBRegressor
xgb1 = XGBRegressor()
cv_score = cross_val_score(xgb1, train_new_1, y_train, cv=4,n_jobs=5)
print(cv_score.mean())

# Gradient Boost Regression and XGBoost Regression will be used for
# further hyperparameter tuning
def model_parameter_tuning(x,y,model,parameters,cross_folds):
    model_grid = GridSearchCV(model,
                        parameters,
                        cv = cross_folds,
                        n_jobs = 5,
                        verbose=True)
    model_grid.fit(x,y)
    y_predicted = model_grid.predict(x)
    print(model_grid.score)
    print(model_grid.best_params_)
    print("The RMSE score is",np.sqrt(np.mean((y-y_predicted)**2)))

#defining function for hyper parameter tuning and using RMSE as my metric
parameters_xgb = {'nthread':[3,4], 
              'learning_rate':[0.02,0.03], #so called `eta` value
              'max_depth': [3,2,4],
              'min_child_weight':[3,4,5],
              'silent': [1],
              'subsample': [0.5],
              'colsample_bytree': [0.7],
              'n_estimators': [300,320]
             }
parameters_gbr={'loss':['ls','lad'],
               'learning_rate':[0.3],
               'n_estimators':[300],
               'min_samples_split':[3,4],
               'max_depth':[3,4],
               'min_samples_leaf':[3,4,2],
               'max_features':['auto','log2','sqrt']
              }

# Defining the useful parameters for parameter tuning
# to get the optimum output
model_parameter_tuning(train_new_1,y_train,xgb1,parameters_xgb,4)

gbr=GradientBoostingRegressor()
model_parameter_tuning(train_new_1,y_train,gbr,parameters_gbr,4)

# Standardization of the model before training
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized=scaler.fit_transform(train_new_1)
column_names = train_new_1.columns
df_standardized = pd.DataFrame(data=standardized,columns=column_names)
df_standardized.head()

basic_model_selection(df_standardized,y_train,4,models)

#Average score for XGBoost matrix
# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=df_standardized,label=y_train)
# import XGBRegressor
xgb1 = XGBRegressor()
cv_score = cross_val_score(xgb1, df_standardized, y_train, cv=4,n_jobs=5)
print(cv_score.mean())

# The Models for hyperparameter tuning are same XGBoost and GradientBoostingRegression
model_parameter_tuning(df_standardized,y_train,xgb1,parameters_xgb,4)

model_parameter_tuning(df_standardized,y_train,gbr,parameters_gbr,4)

# My dataset having outliers make it more prone to mistakes
# Robust Scaler handles the outliers as well
# It scales according to the quartile range

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

normalize = MinMaxScaler()
robust = RobustScaler(quantile_range = (0.1,0.8)) #range of inerquartile is one of the parameters
robust_stan = robust.fit_transform(train_new_1)
robust_stan_normalize = normalize.fit_transform(robust_stan)
# also normalized the dataset using MinMaxScaler i.e has bought the data set between (0,1)
df_robust_normalize = pd.DataFrame(robust_stan_normalize,columns=column_names)
df_robust_normalize.head()

basic_model_selection(df_robust_normalize,y_train,4,models)

cv_score = cross_val_score(xgb1, df_robust_normalize, y_train, cv=4,n_jobs=5)
print(cv_score.mean())

model_parameter_tuning(df_robust_normalize,y_train,xgb1,parameters_xgb,4)
model_parameter_tuning(df_robust_normalize,y_train,gbr,parameters_gbr,4) 


# Best Model
# Comparing all models using RMSE score
# Gradient Boosting Method is the best method when implemented using Robust Scaler and MinMaxScaler normalization

robust_test = robust.fit_transform(test_new)
robust_normalize_test = normalize.fit_transform(robust_test)
df_test_robust_normalize = pd.DataFrame(robust_normalize_test,columns=column_names)

gbr = GradientBoostingRegressor(learning_rate= 0.3, loss= 'lad',max_depth= 3,min_samples_leaf=2,min_samples_split=3
                                ,n_estimators= 300)
# Defining my final model that I will use for prediction

gbr.fit(df_robust_normalize,y_train)

final_prediction=gbr.predict(df_test_robust_normalize) #Predicting the outlet sales

df_final_prediction = pd.DataFrame(final_prediction,columns=['Item_Outlet_Sales'])

df_final_prediction.head()








