# Supress Warnings

import warnings
from IPython.display import display
warnings.filterwarnings('ignore')
import math
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels as sm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Import Data
def importData(path):
    data = pd.read_csv(path)
    return data

# Data Exploration
def dataExploration(data):
    print("Data Head:")
    display(data.head())
    print("=====================================")
    print("Data Info:")
    display(data.info())
    print("=====================================")
    print("Data Describe:")
    display(data.describe())
    print("=====================================")
    print(f"Data shape: {data.shape}")
    print("=====================================")
    print(f"Data columns: {data.columns}")
    print("=====================================")
    print(f"Data types:")
    display(data.dtypes)
    print("=====================================")
  
def dataExplorationTypes(data):
    print(f"Data nulls")
    display(data.isnull().sum())
    print("=====================================")
    print("Checking the percentage of missing values")
    print(round(100*(data.isnull().sum()/len(data.index)), 2))
    print("=====================================")
    print(f"Data not unique")
    display(data.nunique())
    print("=====================================")
    print(f"Data Correlation")
    display(data.corr())
    print("=====================================")
    print(f"Data Skewness")
    display(data.skew())
    print("=====================================")

def dataPercentageNullValues(data, percentage =0):
  # Below code gives list of columns with non 0% value
  null_percentage = round(100*(data.isnull().sum()/len(data.index)), 2)

  # Below code gives list of columns with non 0% value
  col_non_null = null_percentage[null_percentage > percentage]
  return col_non_null

def dataGetValueCountsPct(data):
  for clm in data.columns:
    print(f"Percentage value counts: {clm}")
    print(data[clm].value_counts(normalize=True, ascending=True).mul(100).round(2))

# Data Visualization
def dataVisualization(data, x_vars, y_vars):
    sns.set_style('darkgrid')
    sns.set_palette('Set1')
    plt.figure(figsize = (20,10))        # Size of the figure
    plt.title("Distplot")
    sns.distplot(data[y_vars])
    plt.show()

    plt.figure(figsize = (20,10))        # Size of the figure
    plt.title("Heatmap")
    sns.heatmap(data.corr(), annot=True)
    plt.show()
  
def dataVisualizationRelation(data, x_vars, y_vars):
    sns.set_style('darkgrid')
    sns.set_palette('Set1')
    plt.figure(figsize = (20,10))        # Size of the figure
    sns.pairplot(data,  x_vars=x_vars, y_vars=y_vars)
    plt.show()
    
    for x in x_vars:
      plt.figure(figsize = (20,10))
      plt.title(f"{x} vs {y_vars[0]}")
      sns.regplot(x = x, y=y_vars[0], data=data)
      plt.show()

def dataBoxPlotMultipleVisualization(data, x_vars, y_vars):
  elements = len(x_vars)
  no_cols = 3
  no_rows = math.ceil(elements/no_cols)

  plt.figure(figsize=(30,20))
  for x,i in zip(x_vars, range(elements)):
    plt.subplot(no_rows,no_cols,i+1)
    sns.boxplot(x = x, y = y_vars, data = data)
  plt.show()

def dataBarGraphVisualization(data, x_vars, y_vars):
  plt.figure(figsize=(20,12))
  sns.barplot(x = x_vars, y = y_vars, data = data)
  plt.show()

def dataBoxPlotVisualization(data, x_vars, y_vars):
  plt.figure(figsize=(20,12))
  sns.boxplot(x = x_vars, y = y_vars, data = data)
  plt.show()

def dataPairPlotVisualization(data, x, y):
  sns.set_style('darkgrid')
  sns.set_palette('Set1')
  sns.pairplot(data,  x_vars=x, y_vars=y, size=8)
  plt.show()

def modeling_display(X, y, y_pred):
  sns.set_style('darkgrid')
  sns.set_palette('Set1')
  plt.figure(figsize = (20,10))        # Size of the figure
  plt.scatter(X,y, color='blue')
  plt.plot(X,y_pred, color='red',linewidth=3)
  plt.xlabel("X variable")
  plt.ylabel("Y variable")
  plt.show()

def display_variable_vs_predicted(X,y,y_pred):
    sns.set_style('darkgrid')
    sns.set_palette('Set1')
    for var in X:
      plt.figure(figsize = (20,10))        # Size of the figure
      plt.scatter( X[var] , y , color = 'blue') # original data shown as blue points
      plt.plot(X[var] , y_pred , color = 'red' , linewidth = 3) # Fitted model in red
      plt.xlabel(f"{var}")
      plt.ylabel("Y values")
      plt.show()
    
#Data manupulation
def dataBinarymap(data, varList):
  # Applying the function to the housing list
  data[varList] = data[varList].apply(lambda x: x.astype(str).str.lower())
  data[varList] = data[varList].apply(lambda x: x.map({'yes': 1, "no": 0}))
  return data

def dataGetDummies(data, varList):
  dummies = pd.get_dummies(data, columns=varList, drop_first=True)
  return dummies

# Data splitting
def train_test_data(data):
  # We specify this so that the train and test data set always have the same rows, respectively
  np.random.seed(0)
  df_train,df_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=100)
  return df_train,df_test

def train_test_slr(X, y):
  Xtrain,Xtest,ytrain, ytest = train_test_split(X,y, train_size=0.7, test_size=0.3, random_state=100)
  return Xtrain,Xtest,ytrain, ytest

#Modelling
def train_slr_ols(Xtrain, ytrain):
  print("Starting Simple Linear regression")
  # Add a constant to get an intercept
  X_train_sm = sm.add_constant(Xtrain)

  # Fit the resgression line using 'OLS'
  lr = sm.OLS(ytrain, X_train_sm).fit()

  print(lr.params)
  print(lr.summary())
  y_train_pred = lr.predict(X_train_sm)

  display_variable_vs_predicted(Xtrain,ytrain,y_train_pred)

  return lr, y_train_pred

  
def train_lr_linearRegression(Xtrain, ytrain):
  # Representing LinearRegression as lr(Creating LinearRegression Object)
  X_train_lm = Xtrain.reshape(-1,1)

  lm = LinearRegression()

  # Fit the model using lr.fit()
  lm.fit(X_train_lm, ytrain)

  print(f"Intrtcept: {lm.intercept_}")
  print(f"Coeffecients: {lm.coef_}")

  y_train_pred = lm.predict(X_train_lm)

  return lm, y_train_pred

def train_lr_reshaped(Xtrain, ytrain):
  lm = LinearRegression()

  # Fit the model using lr.fit()
  lm.fit(Xtrain, ytrain)

  print(f"Intrtcept: {lm.intercept_}")
  print(f"Coeffecients: {lm.coef_}")

  y_train_pred = lm.predict(Xtrain)
  return lm, y_train_pred

def train_lr_linearRegression_RFE(Xtrain, ytrain, max_features):
  # Running RFE with the output number of the variable equal to 10
  lm = LinearRegression()
  lm.fit(Xtrain, ytrain)

  rfe = RFE(lm, n_features_to_select=max_features)             # running RFE
  rfe = rfe.fit(Xtrain, ytrain)

  list_df = pd.DataFrame(list(zip(Xtrain.columns,rfe.support_,rfe.ranking_)), columns = ['Features', 'Selection', 'Rank'])
  print(f"Features selected: {list(Xtrain.columns[rfe.support_])}")
  print(f"Features droped: {list(Xtrain.columns[~rfe.support_])}")
  display(list_df)
  return lm, list_df, list(Xtrain.columns[rfe.support_]), list(Xtrain.columns[~rfe.support_])

def train_logistic_model(Xtrain, ytrain,output_var_name):
  # Logistic regression model
  X_train_sm = sm.add_constant(Xtrain)
  logml = sm.GLM(ytrain,X_train_sm, family = sm.families.Binomial())
  res = logml.fit()
  display(res.summary())
  ytrain_pred = res.predict(X_train_sm)
  y_train_pred_df = pd.DataFrame({output_var_name:ytrain.values, f'{output_var_name}_Prob':ytrain_pred})
  # Let's create columns with different probability cutoffs 
  numbers = [float(x)/10 for x in range(10)]
  for i in numbers:
      y_train_pred_df[i]= y_train_pred_df.Churn_Prob.map(lambda x: 1 if x > i else 0)
  y_train_pred_df.head()
  # Let's see the head
  display(y_train_pred_df.head())
  return res, ytrain_pred, y_train_pred_df

#Analysis
def vif_and_pvalue(lr, Xtrain):
  # Create a dataframe that will contain the names of all the feature variables and their respective VIFs
  vif = pd.DataFrame()
  vif['Features'] = Xtrain.columns
  vif['VIF'] = [variance_inflation_factor(Xtrain.values, i) for i in range(Xtrain.shape[1])]
  vif['VIF'] = round(vif['VIF'], 2)
  vif = vif.sort_values(by = "VIF", ascending = False)
  df = pd.DataFrame(lr.pvalues).reset_index()
  df.columns = ['Features', 'p-value']
  vif = vif.set_index('Features').join(df.set_index('Features'))
  return vif

def residual(X_train, ytrain, y_train_pred):
  res = (ytrain - y_train_pred)
  fig = plt.figure()
  sns.distplot(res, bins = 15)
  fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
  plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
  plt.show()

def residual_vs_eachvariable(x, y, y_pred):
  residual = y - y_pred
  for var in x:
    plt.scatter( x[var] , residual)
    plt.axhline(y=0, color='r', linestyle=':')
    plt.xlabel(f"{var}")
    plt.ylabel("Residual")
    plt.show()


return_list = ['prob','True Positve', 'True Negative','False Positve', 'False Negative', 'Accuracy','Sensitivity','Specificity', 'Fasle Positive Rate', 'Positive predicitve value', 'Negative predicitve value','Precision Score','Recall Score','F1 Score']
def logistic_analysis(prob_cutoff, y, y_pred):
  confusion = metrics.confusion_matrix(y, y_pred)
  TP = confusion[1,1] # true positive 
  TN = confusion[0,0] # true negatives
  FP = confusion[0,1] # false positives
  FN = confusion[1,0] # false negatives
  accuracy = metrics.accuracy_score(y, y_pred)
  sensi = TP / float(TP+FN)
  speci = TN / float(TN+FP)
  FPR = FP/ float(TN+FP)
  PPV = TP / float(TP+FP)
  NPV = TN / float(TN+ FN)
  PS = precision_score(y, y_pred)
  RS = recall_score(y, y_pred)
  F1S = metrics.f1_score(y,y_pred)
  return prob_cutoff ,TP, TN, FP,FN, accuracy,sensi,speci,FPR,PPV,NPV, PS, RS, F1S

def logistic_analysis_df(ytrain, ytrain_pred_Prob, ytrain_pred_df):
  # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
  df = pd.DataFrame( columns = return_list)

  num = [float(x)/10 for x in range(10)]
  for i in num:
      df.loc[i] = list(logistic_analysis(i,ytrain, ytrain_pred_df[i]))
  display(df)
  # Let's plot accuracy sensitivity and specificity for various probabilities.
  df.plot.line(x='prob', y=['Accuracy','Sensitivity','Specificity'])
  plt.show()
  p, r, thresholds = precision_recall_curve(ytrain, ytrain_pred_Prob)
  plt.plot(thresholds, p[:-1], "g-")
  plt.plot(thresholds, r[:-1], "r-")
  plt.show()
  return df

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

#Test Data prediction
def test_data_predict(X_test,y_test, lr):
  # Add a constant to X_test
  X_test_sm = sm.add_constant(X_test)

  # Predict the y values corresponding to X_test_sm
  y_pred = lr.predict(X_test_sm)
  modeling_display(X_test, y_test, y_pred)
  return y_pred

#Test data analysis
def test_data_linear_reg_analysis(y, y_pred):
  #Returns the mean squared error; we'll take a square root
  rms = np.sqrt(mean_squared_error(y, y_pred))
  r_squared = r2_score(y, y_pred)
  rss = np.sum(np.square(y - y_pred))
  mse = mean_squared_error(y, y_pred)
  rmse = mse**0.5

  Metrics = ['RMS', 'RSquared', 'RSS', 'MSE', 'RMSE']
  value = [rms, r_squared, rss, mse, rmse]
  test_analysis_df = pd.DataFrame(
    { 'Metrics' : Metrics,
      'Value': value })
  test_analysis_df = test_analysis_df.set_index('Metrics')
  display(test_analysis_df)

  residual = y - y_pred
  plt.scatter( y_pred , residual)
  plt.axhline(y=0, color='r', linestyle=':')
  plt.xlabel("Predicted Values")
  plt.ylabel("Residual")
  plt.show()

  # Distribution of errors
  p = sns.distplot(residual,kde=True)
  p = plt.title('Normality of error terms/residuals')
  plt.xlabel("Residual")
  plt.show()


  return test_analysis_df

def test_data_logistic_reg_analysis(y_test, y_pred_prob, prob_cutoff):
  y_pred_final = y_pred_prob.map(lambda x: 1 if x > 0.42 else 0)
  name = return_list
  value = list(logistic_analysis(prob_cutoff,y_test, y_pred_final))
  test_analysis_df = pd.DataFrame(
    {'Name': name,
     'Value': value
    })
  display(test_analysis_df)
  df = pd.DataFrame( 
    {
      'y_test': y_test,
      'y_test_prob': y_pred_prob,
      'y_test_pred':y_pred_final
    })
  display(df.head())
  return df,test_analysis_df






