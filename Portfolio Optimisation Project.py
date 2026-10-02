#Monte Carlo portfolio optimization through random portfolio generation.



#Suggested 12-stock universe
#Company	NSE Ticker	Sector	Role in portfolio
#HDFC Bank	HDFCBANK	Financials	Banking
#Reliance Industries	RELIANCE	Energy / Telecom / Consumer	Diversified large-cap
#Infosys	INFY	IT	Technology / exports
#Bharti Airtel	BHARTIARTL	Telecom	Communication
#Larsen & Toubro	LT	Industrials / Construction	Infrastructure
#ITC	ITC	FMCG	Consumer / defensive
#Sun Pharmaceutical	SUNPHARMA	Healthcare	Pharmaceuticals
#Mahindra & Mahindra	M&M	Auto	Automotive
#Tata Steel	TATASTEEL	Metals	Commodity/cyclical
#Bharat Electronics	BEL	Capital Goods / Defence	Defence electronics
#NTPC	NTPC	Power	Utilities
#Titan Company	TITAN	Consumer Durables	Discretionary consumer


import pandas as pd
import yfinance as yf
import numpy as np
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.graph_objects as go



tickers = {                                                
    "HDFCBANK": "HDFCBANK.NS",
    "RELIANCE": "RELIANCE.NS",
    "INFY": "INFY.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "LT": "LT.NS",
    "ITC": "ITC.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "M&M": "M&M.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "BEL": "BEL.NS",
    "NTPC": "NTPC.NS",
    "TITAN": "TITAN.NS"}


close_price_df = pd.DataFrame()                                       
for stock_name, ticker in tickers.items():                          
    data = yf.download(ticker,period="5y",auto_adjust=True,progress=False)                    #Auto adjust tells Yahoo Finance to adjust the historical prices for things such as:stock splits , dividends
    close_price_df[stock_name] = data["Close"]   
     
     
close_price_df = close_price_df.dropna()
print("Historical Close Prices are:\n", close_price_df)
daily_returns_df = close_price_df.pct_change() * 100
daily_returns_df.replace(np.nan, 0, inplace = True)
print("Historical Daily Returns are:\n", daily_returns_df.round(3))



def plot_financial_data(df, title):
    fig = px.line(title = title)
    for i in df.columns[:]:
        fig.add_scatter(x=df.index, y=df[i], mode='lines', name=i)
        fig.update_traces(line_width=2)
        fig.update_layout(plot_bgcolor='white')
    fig.show()
plot_financial_data(close_price_df,"Stocks Adjusted Closing Prices")    
plot_financial_data(daily_returns_df,"Stocks Daily Returns")



def price_scaling(raw_prices_df):
    scaled_prices_df = raw_prices_df.copy()
    for i in raw_prices_df.columns[:]:
          scaled_prices_df[i] = raw_prices_df[i]/raw_prices_df[i].iloc[0]
    return scaled_prices_df
print("Scaled Prices are:\n", price_scaling(close_price_df))



#def generate_market_cap_weights(tickers):
 #   market_caps = []
 #   for stock_name, ticker_symbol in tickers.items():
 #       ticker = yf.Ticker(ticker_symbol)
 #       market_cap = ticker.info["marketCap"]
 #       market_caps.append(market_cap)
 #   total_market_cap = np.sum(market_caps)
 #    weights = []
 #   for market_cap in market_caps:
 #       weight = market_cap / total_market_cap
 #       weights.append(weight)
 #   return market_caps, weights
#market_caps, weights = generate_market_cap_weights(tickers)
#print("Market Weights:", weights)
#print("Total Weight:", sum(weights))
#print("Total Weight (%):", sum(weights) * 100)



def generate_portfolio_weights(n):
    weights=[]
    for i in range (n):
        weights.append(random.random())
    weights=weights/np.sum(weights)
    return weights
weights=generate_portfolio_weights(12)
#print(weights)



def asset_allocation(df, weights, initial_investment):
    portfolio_df = df.copy()
    scaled_df = price_scaling(df)                     # Scale stock prices using the "price_scaling" function that we defined earlier (Make them all start at 1)
    for i, stock in enumerate(scaled_df.columns[:]):
        portfolio_df[stock] = scaled_df[stock] * weights[i] * initial_investment         #BUY-AND-HOLD portfolio , Buy at the beginning and hold the stocks without rebalancing.
        
    portfolio_df['Portfolio Value [$]'] = portfolio_df.sum(axis = 1, numeric_only = True)      # Sum up all values and place the result in a new column titled "portfolio value [$]"                
    portfolio_df['Portfolio Daily Return [%]'] = portfolio_df['Portfolio Value [$]'].pct_change(1) * 100              # Calculate the portfolio percentage daily return and replace NaNs with zeros
    portfolio_df.replace(np.nan, 0, inplace = True)
    return portfolio_df
portfolio_df = asset_allocation(close_price_df, weights, 1000000)
#print(portfolio_df.round(2))


#plot_financial_data(portfolio_df[['Portfolio Daily Return [%]']] , 'Portfolio Percentage Daily Return [%]')
#plot_financial_data(portfolio_df.drop(['Portfolio Value [$]', 'Portfolio Daily Return [%]'], axis = 1), 'Portfolio positions [$]')
#plot_financial_data(portfolio_df[['Portfolio Value [$]']], 'Total Portfolio Value [$]')



def simulation_engine(weights, initial_investment):
    portfolio_df = asset_allocation(close_price_df, weights, initial_investment)
    return_on_investment = ((portfolio_df['Portfolio Value [$]'].iloc[-1] - portfolio_df['Portfolio Value [$]'].iloc[0])/ 
                             portfolio_df['Portfolio Value [$]'].iloc[0]) * 100
    portfolio_daily_return_df = portfolio_df.drop(columns = ['Portfolio Value [$]', 'Portfolio Daily Return [%]'])
    portfolio_daily_return_df = portfolio_daily_return_df.pct_change(1) 
    expected_portfolio_return = np.sum(weights * portfolio_daily_return_df.mean() ) * 252
    covariance = portfolio_daily_return_df.cov() * 252 
    expected_volatility = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
    rf = 0.03                                                                                                     #annual risk free rate of return (3% in this case)
    sharpe_ratio = (expected_portfolio_return - rf)/expected_volatility 
    return expected_portfolio_return, expected_volatility, sharpe_ratio, portfolio_df['Portfolio Value [$]'].iloc[-1], return_on_investment

initial_investment = 1000000
portfolio_metrics = simulation_engine(weights, initial_investment)

#print('Expected Portfolio Annual Return = {:.2f}%'.format(portfolio_metrics[0] * 100))                           #.2f is a formatting instruction in Python.It means: Show the number as a floating-point number with 2 digits after the decimal point.
#print('Portfolio Standard Deviation (Volatility) = {:.2f}%'.format(portfolio_metrics[1] * 100))
#print('Sharpe Ratio = {:.2f}'.format(portfolio_metrics[2]))
#print('Portfolio Final Value = ${:.2f}'.format(portfolio_metrics[3]))
#print('Return on Investment = {:.2f}%'.format(portfolio_metrics[4]))



#SIMULATION ENGINE
n = len(close_price_df.columns)
sim_runs = 10000
weights_runs = np.zeros((sim_runs, n))
sharpe_ratio_runs = np.zeros(sim_runs)
expected_portfolio_returns_runs = np.zeros(sim_runs)
volatility_runs = np.zeros(sim_runs)
return_on_investment_runs = np.zeros(sim_runs)
final_value_runs = np.zeros(sim_runs)

for i in range(sim_runs):
    weights = generate_portfolio_weights(n)
    weights_runs[i,:] = weights
    expected_portfolio_returns_runs[i], volatility_runs[i], sharpe_ratio_runs[i], final_value_runs[i], return_on_investment_runs[i] = simulation_engine(weights, initial_investment)
    #print("Simulation Run = {}".format(i))   
    #print("Weights = {}, Final Value = ${:.2f}, Sharpe Ratio = {:.2f}".format(weights_runs[i].round(3), final_value_runs[i], sharpe_ratio_runs[i]))   
    #print('\n')



#print(sharpe_ratio_runs)
print('Index of the maximum Sharpe ratio:', sharpe_ratio_runs.argmax()) #returns the index of the maximum Sharpe ratio value
print('Maximum Sharpe ratio:', sharpe_ratio_runs.max())
#print(weights_runs)
# Obtain the portfolio weights that correspond to the maximum Sharpe ratio 
print('Portfolio weights corresponding to the maximum Sharpe ratio:', weights_runs[sharpe_ratio_runs.argmax(), :])
#Return Sharpe ratio, volatility corresponding to the best weights allocation (maximum Sharpe ratio)
optimal_portfolio_return, optimal_volatility, optimal_sharpe_ratio, highest_final_value, optimal_return_on_investment = simulation_engine(weights_runs[sharpe_ratio_runs.argmax(), :], initial_investment)
print('\n')
print('Best Portfolio Metrics Based on {} Monte Carlo Simulation Runs:'.format(sim_runs))
print('  - Portfolio Expected Annual Return = {:.02f}%'.format(optimal_portfolio_return * 100))
print('  - Portfolio Standard Deviation (Volatility) = {:.02f}%'.format(optimal_volatility * 100))
print('  - Sharpe Ratio = {:.02f}'.format(optimal_sharpe_ratio))
print('  - Final Value = ${:.02f}'.format(highest_final_value))
print('  - Return on Investment = {:.02f}%'.format(optimal_return_on_investment))


# tolist is used to convert the numpy arrays to lists, which can be used to create a DataFrame
# Create a DataFrame that contains volatility, return, and Sharpe ratio for all simualation runs
sim_out_df = pd.DataFrame({'Volatility': volatility_runs.tolist(), 'Portfolio_Return': expected_portfolio_returns_runs.tolist(), 'Sharpe_Ratio': sharpe_ratio_runs.tolist() })
#print(sim_out_df)


# Plot volatility vs. return for all simulation runs
# Highlight the volatility and return that corresponds to the highest Sharpe ratio
fig = px.scatter(sim_out_df, x = 'Volatility', y = 'Portfolio_Return', color = 'Sharpe_Ratio', size = 'Sharpe_Ratio', hover_data = ['Sharpe_Ratio'] )
fig.update_layout({'plot_bgcolor': "white"})
fig.show()


  
nifty_data = yf.download("^NSEI",period="5y",auto_adjust=True,progress=False)
nifty_close = nifty_data["Close"].squeeze()
nifty_returns = nifty_close.pct_change().dropna()
nifty_total_return = (nifty_close.iloc[-1] / nifty_close.iloc[0] - 1) * 100
years = (nifty_close.index[-1] - nifty_close.index[0]).days / 365.25
nifty_cagr = ((nifty_close.iloc[-1] / nifty_close.iloc[0]) ** (1 / years) - 1) * 100 
nifty_volatility = nifty_returns.std() * np.sqrt(252) * 100
risk_free_rate = 0.03
nifty_annual_return = nifty_returns.mean() * 252
nifty_sharpe = (nifty_annual_return - risk_free_rate) / (nifty_returns.std() * np.sqrt(252))
nifty_portfolio_value = (nifty_close / nifty_close.iloc[0]) * initial_investment



print("\nBenchmark Comparison")
print("--------------------")

print("Optimized Portfolio Expected Return = {:.2f}%".format(optimal_portfolio_return * 100))
print("NIFTY 50 Annualized Return = {:.2f}%".format(nifty_annual_return * 100))

print("Optimized Portfolio Volatility = {:.2f}%".format(optimal_volatility * 100))
print("NIFTY 50 Volatility = {:.2f}%".format(nifty_volatility))

print("Optimized Portfolio Sharpe = {:.2f}".format(optimal_sharpe_ratio))
print("NIFTY 50 Sharpe = {:.2f}".format(nifty_sharpe))



comparison_df = pd.DataFrame({"Optimized Portfolio": portfolio_df["Portfolio Value [$]"],"NIFTY 50": nifty_portfolio_value})
fig = px.line(comparison_df,x=comparison_df.index,y=["Optimized Portfolio", "NIFTY 50"],title="Optimized Portfolio vs NIFTY 50")
fig.update_layout(yaxis_title="Portfolio Value (₹)",xaxis_title="Date",plot_bgcolor="white")
fig.show()