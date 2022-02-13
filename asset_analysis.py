import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, fmin, fmin_l_bfgs_b
import scipy.stats
import bt

bitcoin = pd.read_csv('./data/bitcoin-usd.csv', parse_dates=['date'])
bitcoin.head()

sp500 = pd.read_csv('./data/sp500.csv', parse_dates=['date'])
sp500.head()

monthly_data = pd.read_csv('./data/monthly_data.csv', parse_dates=['date'])
monthly_data.head()

# Part1 Compare the performance of Bitcoin to S&P 500 and Gold
## Plot price trends
plt.figure(figsize = (8,6))
plt.plot(bitcoin.date,bitcoin.close, label = 'bitcoin')
plt.plot(sp500.date,sp500.close, label = 'sp500')
plt.plot(monthly_data.date, monthly_data.gold_usd, label = 'gold')
plt.legend()
plt.title('price trend comparison')
plt.show()

## Calculate asset returns
bitcoin['returns'] = bitcoin.close.pct_change()
sp500['returns'] = sp500.close.pct_change()
monthly_data['gold_returns'] = monthly_data['gold_usd'].pct_change()
## Extract the month and year
monthly_data['date'] = pd.to_datetime(monthly_data['date'],format='%Y%m%d')
monthly_data['year'] = pd.DatetimeIndex(monthly_data['date']).year
monthly_data['month'] = pd.DatetimeIndex(monthly_data['date']).month
bitcoin['date'] = pd.to_datetime(bitcoin['date'],format='%Y%m%d')
bitcoin['year'] = pd.DatetimeIndex(bitcoin['date']).year
bitcoin['month'] = pd.DatetimeIndex(bitcoin['date']).month
sp500['date'] = pd.to_datetime(sp500['date'],format='%Y%m%d')
sp500['year'] = pd.DatetimeIndex(sp500['date']).year
sp500['month'] = pd.DatetimeIndex(sp500['date']).month
## Calculate monthly returns
bitcoin_month = bitcoin.groupby(['year','month'])['returns'].sum()
bitcoin_month= pd.DataFrame(bitcoin_month)
sp500_month = sp500.groupby(['year','month'])['returns'].sum()
sp500_month= pd.DataFrame(sp500_month)
gold_month = pd.DataFrame(monthly_data.groupby(['year','month'])['gold_returns'].sum())
monthly_prices = bitcoin_month.merge(sp500_month, left_index = True,right_index=True, suffixes = ['_btc', '_sp500'])
monthly_prices = monthly_prices.merge(gold_month, left_index = True,right_index=True)

## Plot return trends
plt.figure(figsize = (8,6))
plt.plot(monthly_prices['returns_btc'].values, label = 'bitcoin')
plt.plot(monthly_prices['returns_sp500'].values, label = 'sp500')
plt.plot(monthly_prices['gold_returns'].values, label = 'gold')
plt.legend()
plt.title('monthly returns trend comparison')
plt.show()

# Plot the histogram
monthly_prices['returns_btc'].hist(bins = 100, color='red')
plt.ylabel('Frequency')
plt.xlabel('Monthly return')
plt.title('Bitcoin Monthly Return Histogram')
plt.show()

monthly_prices['returns_sp500'].hist(bins = 100, color='blue')
plt.ylabel('Frequency')
plt.xlabel('Monthly return')
plt.title('S&P500 Monthly Return Histogram')
plt.show()

monthly_prices['gold_returns'].hist(bins = 100, color='green')
plt.ylabel('Frequency')
plt.xlabel('Monthly return')
plt.title('Gold Monthly Return Histogram')
plt.show()

# Part2 Bitcoin being used as a hedge versus inflation

## Calculate annual inflation rate
year_data = monthly_data.groupby(['year'])['cpi_us'].mean()
year_data = pd.DataFrame(year_data)
year_data['inflation'] = year_data['cpi_us'].pct_change()
## Inflation Trend
plt.figure(figsize = (10,8))
plt.plot(year_data.index, year_data.inflation, label = 'inflation rate')
plt.title('Annual Inflation Rate')
plt.legend()
plt.show()

print('highest inflation rate:', year_data.inflation.max(),
     '\nlowest inflation rate:', year_data.inflation.min())

## Calculate annual bitcoin returns
bitcoin_year = bitcoin.groupby(['year'])['close'].mean()
bitcoin_year = pd.DataFrame(bitcoin_year )
bitcoin_year['returns'] = bitcoin_year['close'].pct_change()
## Calculate annual sp500 returns
sp500_year = sp500.groupby(['year'])['close'].mean()
sp500_year = pd.DataFrame(sp500_year )
sp500_year['returns'] = sp500_year['close'].pct_change()
## Calculate annual gold returns
gold_year =monthly_data.groupby('year').mean().pct_change()['gold_usd']
## Returns and Inflation Trend
plt.figure(figsize = (10,8))
plt.plot(year_data.index, year_data.inflation, label = 'inflation rate')
plt.plot(bitcoin_year.index, bitcoin_year.returns, label = 'BTC returns')
plt.plot(sp500_year.index, sp500_year.returns, label = 's&p500 returns')
plt.plot(gold_year.index, gold_year.values, label = 'gold returns')
plt.title('Annual Inflation Rate and Returns')
plt.legend()
plt.show()

# Part3 Portfolio that minimizes overall risk
bitcoin_year_return = bitcoin_year.returns.mean()
sp500_year_return = sp500_year.returns.mean()
gold_year_return = np.nanmean(gold_year.values)
def risk_p(x):
    if 1-x[0]-x[1]>0:
        x3 = 1-x[0]-x[1]
    else:
        x3 = 0
    return var_sp500_month*(x[0]**2)+var_bitcoin_month*(x[1]**2)+var_gold_month*(x3**2)+2*x[0]*x[1]*corr_BTC_sp500*sd_sp500_month*sd_BTC_month+2*x[0]*x3*corr_gold_sp500*sd_sp500_month*sd_gold_month+2*x[1]*x3*corr_gold_BTC*sd_gold_month*sd_BTC_month


def portfolio_return(x):
	if 1 - x[0] - x[1] > 0:
		x3 = 1 - x[0] - x[1]
	else:
		x3 = 0
	return x[0] * sp500_year_return + x[1] * bitcoin_year_return + x3 * gold_year_return


def max_min_norm(value, max_n, min_n):
	return (value - min_n) / (max_n - min_n)


## Calculate the variance of each asset
var_bitcoin_month = np.var ( bitcoin_month.returns )
var_sp500_month = np.var ( sp500_month.returns )
var_gold_month = np.var ( gold_month.gold_returns )
print ( 'variance of bitcoin in monthly return: ', var_bitcoin_month,
        '\nvariance of s&p500 in monthly return: ', var_sp500_month,
        '\nvariance of gold in monthly return: ', var_gold_month )

## Calculate the standard deviation of each asset
sd_BTC_month = np.sqrt ( var_bitcoin_month )
sd_sp500_month = np.sqrt ( var_sp500_month )
sd_gold_month = np.sqrt ( var_gold_month )
print ( 'standard deviation of bitcoin in monthly return: ', sd_BTC_month,
        '\nstandard deviation of s&p500 in monthly return: ', sd_sp500_month,
        '\nstandard deviation of gold in monthly return: ', sd_gold_month )

## Calculate the correlation between each asset

corr_BTC_sp500 = scipy.stats.pearsonr ( bitcoin_month.returns, sp500_month.returns )[0]
corr_gold_sp500 = scipy.stats.pearsonr ( gold_month.gold_returns, sp500_month.returns )[0]
corr_gold_BTC = scipy.stats.pearsonr ( gold_month.gold_returns, bitcoin_month.returns )[0]
print ( 'correlation of bitcoin and s&p500 in monthly return: ', corr_BTC_sp500,
        '\ncorrelation of s&p500 and gold in monthly return: ', corr_gold_sp500,
        '\ncorrelation of gold and bitcoin in monthly return: ', corr_gold_BTC )

var_portfolio= []
return_portfolio = []
sp500_weigh = []
btc_weigh = []
all_w = []
w_sp500 = np.arange(0,1.1,0.1)
w_btc = np.arange(0,1.1,0.1)
for w1 in w_sp500:
    for w2 in w_btc:
        if w1+w2>1:
            pass
        else:
            x = (w1,w2)
            sp500_weigh.append(w1)
            btc_weigh.append(w2)
            all_w.append(("{:.1f}".format(w1),"{:.1f}".format(w2)))
            var_portfolio.append(risk_p(x))
            return_portfolio.append(portfolio_return(x))

weigh_result = pd.DataFrame({'sp500':sp500_weigh, 'btc':btc_weigh, 'risk':var_portfolio, 'return':return_portfolio, 'all_w': all_w})
weigh_result['risk_norm']=weigh_result.apply(lambda x:max_min_norm(x['risk'], max(weigh_result['risk']), min(weigh_result['risk'])),axis = 1)
weigh_result['return_norm']=weigh_result.apply(lambda x:max_min_norm(x['return'], max(weigh_result['return']), min(weigh_result['return'])),axis = 1)

plt.figure(figsize=(10,8))
plt.plot(weigh_result.sp500, weigh_result.risk_norm, label = 'risk_normalized')
plt.plot(weigh_result.sp500, weigh_result['return_norm'], label = 'return_normalized')

for i in range(0,len(weigh_result)):

    label = str(weigh_result.all_w.iloc[i])

    plt.annotate(label, # this is the text
                 (weigh_result.sp500.iloc[i],weigh_result.risk_norm.iloc[i]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('weigh in sp500')

plt.legend()
plt.show()

x0 = [0.3,0.5]# random initial guess
res = minimize(risk_p, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})

print('Percentage of s&p500: ', res.x[0],
      '\nPercentage of bitcoin: ', res.x[1],
      '\nPercentage of gold: ', 1-res.x[0]-res.x[1])

# Part4 Run backtest to Compare strategies

tickers = {
'stocks':['SPY','GLD','BTC-USD'],

}
prices = bt.data.get(tickers['stocks'], clean_tickers=False)
prices = prices.rename(columns = {'BTC-USD':'BTC'})

strategy1 = bt.Strategy('strategy1',
 algos = [
 bt.algos.RunMonthly(),
 bt.algos.SelectAll(),
 bt.algos.WeighSpecified(SPY=res.x[0],BTC=res.x[1], GLD = 1-res.x[0]-res.x[1]),
 bt.algos.Rebalance(),
 ]
)
strategy_equalweigh = bt.Strategy('strategy_equalweigh',
 algos = [
 bt.algos.RunMonthly(),
 bt.algos.SelectAll(),
 bt.algos.WeighEqually(),
 bt.algos.Rebalance(),
 ]
)



strategy_sp500 = bt.Strategy('strategy_sp500',
 algos = [bt.algos.RunMonthly(),
 bt.algos.SelectAll(),
 bt.algos.SelectThese(['SPY']),
 bt.algos.WeighEqually(),
 bt.algos.Rebalance()],
)

strategy_gold = bt.Strategy('strategy_gold',
 algos = [bt.algos.RunMonthly(),
 bt.algos.SelectAll(),
 bt.algos.SelectThese(['GLD']),
 bt.algos.WeighEqually(),
 bt.algos.Rebalance()],
)

strategy_btc = bt.Strategy('strategy_btc',
 algos = [bt.algos.RunMonthly(),
 bt.algos.SelectAll(),
 bt.algos.SelectThese(['BTC']),
 bt.algos.WeighEqually(),
 bt.algos.Rebalance()],
)

backtest_s1 = bt.Backtest(strategy1,prices)
backtest_equalweigh = bt.Backtest(strategy_equalweigh,prices)
backtest_sp500 = bt.Backtest(strategy_sp500, prices)
backtest_gold = bt.Backtest(strategy_gold, prices)
backtest_btc = bt.Backtest(strategy_btc, prices)

report = bt.run(backtest_s1, backtest_equalweigh,backtest_sp500, backtest_gold, backtest_btc)

plt.figure()
report.plot()
plt.legend()
plt.show()

report.display()