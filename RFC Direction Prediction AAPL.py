import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


#########In the paper, the training data consists of 30 days,
#and the time horizons they're predicting on are 1 month, 2 months and 3 months. I'm training the model on 70%
#of data from 2019/01/01, testing it on the remaining 30%, with a time horizon of 90 days.
apl=yf.Ticker("AAPL")
#apl=apl.history(period="max", auto_adjust=False)
apl=apl.history(start="2019-01-01", end=dt.today().strftime('%Y-%m-%d'), auto_adjust=False)
apl=apl.drop(columns=["Dividends", "Stock Splits"])

##########Exponential Smoothing - exponentially decreasing weights to past observations, recent
#observations weighted higher
alpha=0.7 #Can play with this; large alpha reduces the smoothing
apl["Smoothed_Close"]=alpha*apl["Close"]+(1-alpha)*(apl["Close"].shift(1))
apl["Smoothed_Open"]=alpha*apl["Open"]+(1-alpha)*(apl["Open"].shift(1))
apl["Smoothed_High"]=alpha*apl["High"]+(1-alpha)*(apl["High"].shift(1))
apl["Smoothed_Low"]=alpha*apl["Low"]+(1-alpha)*(apl["Low"].shift(1))
apl["Smoothed_Volume"]=alpha*apl["Volume"]+(1-alpha)*(apl["Volume"].shift(1))
#Target setup (Equation (3))
d=90 #Time horizon in days
apl["Target"]=np.sign(apl["Smoothed_Close"].shift(-d)-apl["Smoothed_Close"])

plt.plot(apl.index, apl["Smoothed_Close"])
plt.xlabel("Year")
plt.ylabel("Close Price (Smoothed)")
plt.show()

##########Feature Extraction (Defining Technical Indicators)
#RSI
n=d #Set n equal to the time horizon above
U=[0]
D=[0]
for i in range(1,len(apl.Smoothed_Close)):
    if apl.Smoothed_Close[i] > apl.Smoothed_Close[i-1]:
        U.append(apl.Smoothed_Close[i] - apl.Smoothed_Close[i-1])
        D.append(0)
    elif apl.Smoothed_Close[i] < apl.Smoothed_Close[i-1]:
        U.append(0)
        D.append(apl.Smoothed_Close[i-1] - apl.Smoothed_Close[i])
    else:
        U.append(0)
        D.append(0)
apl["Up"]=U
apl["Down"]=D
apl["RS"]=apl["Up"].rolling(n).mean()/apl["Down"].rolling(n).mean()
apl["RSI"]=100-(100)/(1+apl["RS"])
apl=apl.drop(columns=["Up","Down","RS"])

#Stochastic Oscillator %K (Equation (6))
apl["Stoch. Osc."]=100*(apl["Smoothed_Close"]-apl["Smoothed_Close"].rolling(14).min())/(apl["Smoothed_Close"].rolling(14).max()-apl["Smoothed_Close"].rolling(14).min())

#Williams %R (Equation (7))
apl["Williams %R"]=-100*(apl["Smoothed_Close"].rolling(14).max()-apl["Smoothed_Close"])/(apl["Smoothed_Close"].rolling(14).max()-apl["Smoothed_Close"].rolling(14).min())

#Moving Average Convergence Difference (Equations (8) & (9))
apl["MACD"]=apl["Smoothed_Close"].ewm(span=12, adjust=False).mean()-apl["Smoothed_Close"].ewm(span=26, adjust=False).mean()
apl["MACD Signal Line"]=apl["MACD"].ewm(span=9, adjust=False).mean()

#Price Rate of Change (Equation (10))
apl["PROC"]=(apl["Smoothed_Close"]-apl["Smoothed_Close"].shift(n))/apl["Smoothed_Close"].shift(n)

#On Balance Volume (Equation (11))
OBV=[0]
for i in range(1,len(apl.Smoothed_Close)):
    if apl.Smoothed_Close[i] > apl.Smoothed_Close[i-1]:
        OBV.append(OBV[i-1]+apl.Smoothed_Volume[i])
    elif apl.Smoothed_Close[i] < apl.Smoothed_Close[i-1]:
        OBV.append(OBV[i-1]-apl.Smoothed_Volume[i])
    else:
        OBV.append(apl.Smoothed_Volume[i])
apl["OBV"]=OBV

######And some extra features not in the paper:
#Garmann-Klass Volatility
apl["GKV"]=((np.log(apl["High"])-np.log(apl["Low"]))**2)/2-(2*np.log(2)-1)*(np.log(apl["Close"])-np.log(apl["Open"]))**2

#Open/Close Ratio
apl["OCR"]=apl["Smoothed_Open"]/apl["Smoothed_Close"]

#High/Close Ratio
apl["HCR"]=apl["Smoothed_High"]/apl["Smoothed_Close"]

#Low/Close Ratio
apl["LCR"]=apl["Smoothed_Low"]/apl["Smoothed_Close"]

#High/Low Ratio
apl["HLR"]=apl["Smoothed_High"]/apl["Smoothed_Low"]

#Drop all of the columns in the dataframe we're not going to use as features, & drop NaN columns
apl=apl.drop(columns=["Open", "High", "Low", "Close", "Volume", "Smoothed_Close", "Smoothed_Open", "Smoothed_Volume", "Smoothed_High", "Smoothed_Low"])
apl=apl.dropna(axis=0)

##########Building the RFC model
X_train, X_test, y_train, y_test=train_test_split(apl.drop(columns=["Target"]), apl["Target"], test_size=0.3, random_state=42, shuffle=False) #trainset is the first 75% of the data, testset is the rest


#param_grid_rand = { #Do a random search to get optimal hyperparameters given these features
#    'n_estimators': [65, 80, 95, 110, 125, 140, 155, 170], 
#    'min_samples_split': [10, 20, 30, 40],
#    'max_depth': [12, 20, 28, 36], 
#    'max_leaf_nodes': [150, 200, 250, 300, 350], 
#} 
#rand_search=RandomizedSearchCV(RandomForestClassifier(), param_grid_rand)
#rand_search.fit(X_train, y_train) 
#print(rand_search.best_estimator_)

#param_grid = { #Do a grid search to narrow down further
#    'n_estimators': [110, 125, 140, 155], 
#    'min_samples_split': [30, 40, 50],
#    'max_depth': [12, 20, 28, 36], 
#    'max_leaf_nodes': [150, 200, 250], 
#} 
#grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1) 
#grid_search.fit(X_train, y_train) 
#print(grid_search.best_estimator_) 

#Run RFC with the hyperparams from the grid search, and visualise the feature importance
RFC=RandomForestClassifier(n_estimators=110, criterion="gini", min_samples_split=50, max_depth=20, max_leaf_nodes=250, random_state=42)
RFC.fit(X_train, y_train)
feature_scores=pd.Series(RFC.feature_importances_, index=X_train.columns).sort_values(ascending=False)
predictions=RFC.predict(X_test)
predictions=pd.Series(predictions, index=X_test.index, name="Prediction")#Generating a pandas series of predictions, just for readability
targetpred=pd.concat([predictions, y_test], axis=1)
print(feature_scores, precision_score(y_test, predictions), accuracy_score(y_test, predictions))

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.show()

X_train=X_train.drop(columns=["Williams %R"])#Drop the least important feature(s)
X_test=X_test.drop(columns=["Williams %R"])

#Run again and readjust hyperparams
RFC=RandomForestClassifier(n_estimators=110, criterion="entropy", max_features="sqrt", min_samples_split=31, max_depth=20, max_leaf_nodes=250, random_state=42)
RFC.fit(X_train, y_train)
feature_scores=pd.Series(RFC.feature_importances_, index=X_train.columns).sort_values(ascending=False)
predictions=RFC.predict(X_test)
predictions=pd.Series(predictions, index=X_test.index, name="Prediction")#Generating a pandas series of predictions, just for readability
targetpred=pd.concat([predictions, y_test], axis=1)
print(feature_scores, precision_score(y_test, predictions), accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))