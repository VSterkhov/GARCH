import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy.stats import boxcox
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

################
 # Dickey-Fuller
##################
def test_stationarity(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for [key, value] in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def tsplot(y, lags=None, figsize=(14, 8), style='bmh'):
    test_stationarity(y)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        plt.figure(figsize=figsize)
        layout = (5, 1)
        ts_ax = plt.subplot2grid(layout, (0, 0), rowspan=2)
        acf_ax = plt.subplot2grid(layout, (2, 0))
        pacf_ax = plt.subplot2grid(layout, (3, 0))
        qq_ax = plt.subplot2grid(layout, (4, 0))

        y.plot(ax=ts_ax, color='blue', label='Or')
        ts_ax.set_title('Original')

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        
        plt.tight_layout()
    return

def _get_best_model(TS):
    best_aic = np.inf 
    best_order = None
    best_mdl = None

    for i in range(5):
        for d in range(5):
            for j in range(5):
                try:
                    tmp_mdl = smt.ARIMA(X, order=(i,d,j)).fit(method='innovations_mle')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))                    
    return best_aic, best_order, best_mdl


series = pd.read_csv("./weekly-closings-of-the-dowjones-.csv", header=0, parse_dates=[0], index_col=0, squeeze=True)

plt.figure(figsize=(14,8))
plt.plot(series, color='blue', label='Dowjones closes')
plt.legend()
plt.show()

series = series.diff(periods=-1)
series = series.truncate(after=series.last_valid_index())

garch_model = arch_model(series)
res = garch_model.fit(update_freq=5)

tsplot(res.resid)
fig = res.plot()


fig, ax = plt.subplots(1, 1)
var_2016 = res._volatility


#subplot = var_2016.plot(ax=ax, title="Conditional Variance")
#subplot.set_xlim(var_2016.index[0], var_2016.index[-1])


'''
sim_mod = arch_model(None)#, p=1, o=1, q=1, dist="skewt")
sim_data = sim_mod.simulate(res.params, len(X))
plt.figure(figsize=(14,8))
plt.plot(sim_data['data'], color='Green', label='Simulate')
plt.legend()
plt.show()

#garch_forecast = garch_model.forecast(X, horizon=1)


fig, ax = plt.subplots(1, 1)
var_2016 = res.conditional_volatility["2016"] ** 2.0
subplot = var_2016.plot(ax=ax, title="Conditional Variance")
subplot.set_xlim(var_2016.index[0], var_2016.index[-1])

forecasts = res.forecast(reindex=False)
print(forecasts.residual_variance.iloc[-3:])
#forecasts.variance[split_date:].plot()
'''
'''
aic, order, mdl = _get_best_model(dowjones_closing['Close'])


plt.figure(figsize=(14,8))
ax = plt.axes()
predictions = mdl.predict(1, len(X)+20, ax=ax)
plt.plot(predictions, color='Orange', label='ARIMA')
#plt.plot(dowjones_closing['Close'], color='blue', label='X')
plt.legend()
plt.show()

'''




