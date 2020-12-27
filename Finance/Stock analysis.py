# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:00:49 2019

@author: ESG13
"""
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

#Stock Data
start = datetime.datetime(2009, 1, 1)
end = datetime.datetime(2020, 5, 4)
df = web.DataReader("RTRX", 'yahoo', start, end)
df1 = web.DataReader("GILD", 'yahoo', start, end)
df2 = web.DataReader("NVS", 'yahoo', start, end)
df.tail()
df.to_excel(r'C:\Users\ESG13\Desktop\RTRX.xlsx')
df1.to_excel(r'C:\Users\ESG13\Desktop\GILD.xlsx')
df2.to_excel(r'C:\Users\ESG13\Desktop\NVS.xlsx')
print(df, df1, df2)

#Moving Average
def Movingavg(x,y,z):
    close_px = x['Adj Close']
    mavg = close_px.rolling(window=100).mean()

    import matplotlib.pyplot as plt
    from matplotlib import style

    close_px = x['Adj Close']
    mavg = close_px.rolling(window=100).mean()

    # Adjusting the size of matplotlib
    import matplotlib as mpl
    mpl.rc('figure', figsize=(8, 7))
    mpl.__version__

    # Adjusting the style of matplotlib
    style.use('ggplot')

    close_px.plot(label=y)
    mavg.plot(label='mavg')
    plt.legend()
    print(mavg)
    mavg.to_excel(z)
Movingavg(df,'RTRX',r'C:\Users\ESG13\Desktop\RTRX_Mavg.xlsx')
Movingavg(df1,'GILD',r'C:\Users\ESG13\Desktop\GILD_Mavg.xlsx')
Movingavg(df2,'NVS',r'C:\Users\ESG13\Desktop\NVS_Mavg.xlsx')

#Analyzing Competitor Stocks
dfcomp = web.DataReader(['RTRX', 'GILD', 'NVS'],'yahoo',start=start,end=end)['Adj Close']
print(dfcomp)
dfcomp.to_excel(r'C:\Users\ESG13\Desktop\CompetitorData.xlsx')

#Correlation Matrix
retscomp = dfcomp.pct_change()
corr = retscomp.corr()
print(corr)
corr.to_excel(r'C:\Users\ESG13\Desktop\CorrelationMatrix.xlsx')

"""
#Put each sequences of code below in Ipython to avoid chart interference
#Kernal Density Estimation and Scatter Matrix 
#From here we can see most of the distributions ammond stocks
#Some are positively correlated which will show in a heat map
#(Commented out not to interfere with graph above, run in Ipython to verify results)
pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10,10));

#Heatmap 
#Shows how stocks affect eachother white top being the most (white) and bottom (black) being the least
import matplotlib.pyplot as plt
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);

#Stocks Return Rate and Risk
#top sell bottom buy
import matplotlib.pyplot as plt
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
"""
    
#Feature Engineering forcasting using
#Predicting stocks
#Simple Linear Analysis, Quadratic Discriminant Analysis (QDA), and K Nearest Neighbor (KNN)
def my_func(q,w):   
     dfreg = q.loc[:,['Adj Close','Volume']]
     dfreg['HL_PCT'] = (q['High'] - q['Low']) / q['Close'] * 100.0
     dfreg['PCT_change'] = (q['Close'] - q['Open']) / q['Open'] * 100.0
     import math
     import matplotlib.pyplot as plt
     import numpy as np
     import sklearn
     from sklearn import preprocessing
     from sklearn.linear_model import LinearRegression
     from sklearn.neighbors import KNeighborsRegressor
     from sklearn.linear_model import Ridge
     from sklearn.preprocessing import PolynomialFeatures
     from sklearn.pipeline import make_pipeline
     from sklearn.model_selection import train_test_split
     from sklearn.pipeline import make_pipeline
     # Drop missing value
     dfreg.fillna(value=-99999, inplace=True)
     # We want to separate 1 percent of the data to forecast
     forecast_out = int(math.ceil(0.01 * len(dfreg)))
     # Separating the label here, we want to predict the AdjClose
     forecast_col = 'Adj Close'
     dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
     X = np.array(dfreg.drop(['label'], 1))
     # Scale the X so that everyone can have the same distribution for linear regression
     X = preprocessing.scale(X)
     # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
     X_lately = X[-forecast_out:]
     X = X[:-forecast_out]
     # Separate label and identify it as y
     y = np.array(dfreg['label'])
     y = y[:-forecast_out]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
     len(X)
     # Linear regression
     clfreg = LinearRegression(n_jobs=-1)
     clfreg.fit(X_train, y_train)
     # Quadratic Regression 2
     clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
     clfpoly2.fit(X_train, y_train)
     # Quadratic Regression 3
     clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
     clfpoly3.fit(X_train, y_train)
     #Pipeline(memory=None, steps=[('polynomialfeatures', PolynomialFeatures(degree=3, include_bias=True, interaction_only=False, order='C')), ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001))], verbose=False)
     # KNN Regression
     clfknn = KNeighborsRegressor(n_neighbors=2)
     clfknn.fit(X_train, y_train)
     KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None, n_neighbors=2, p=2,weights='uniform')
     confidencereg = clfreg.score(X_test, y_test)
     confidencepoly2 = clfpoly2.score(X_test,y_test)
     confidencepoly3 = clfpoly3.score(X_test,y_test)
     confidenceknn = clfknn.score(X_test, y_test)
     print('confidencereg')
     print(confidencereg)
     print('confidencepoly2')
     print(confidencepoly2)
     print('confidencepoly3')
     print(confidencepoly3)
     print('confidenceknn')
     print(confidenceknn)
     print(w)
     forecast_set = clfreg.predict(X_lately)
     dfreg['Forecast'] = np.nan
     last_date = dfreg.iloc[-1].name
     last_unix = last_date
     next_unix = last_unix + datetime.timedelta(days=1)
     for i in forecast_set:
         next_date = next_unix
         next_unix += datetime.timedelta(days=1)
         dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
     dfreg['Adj Close'].tail(500).plot()
     dfreg['Forecast'].tail(500).plot()
     plt.legend(loc=4)
     plt.xlabel('Date')
     plt.ylabel('Price')
     plt.show()

#Type this code in IPython
"""
my_func(df,'RTRX')
my_func(df1,'GILD')
my_func(df2,'NVS')
"""

