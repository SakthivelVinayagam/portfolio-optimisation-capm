import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as sco
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import statsmodels.api as sm

# Parameters
# Stocks we want to analyse
STOCKS = ['AAPL', 'INTC', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'IBM', 'META', 'QCOM', 'TM']  # Example stocks
MARKET_INDEX = '^GSPC'   # The S&P 500 is used as the market benchmark
RISK_FREE_RATE = 0.02   # A fixed risk-free rate for calculations
start_date = "2020-03-02"
end_date = "2025-03-02"

# 1. Data Collection
# Download Historical Data.
def get_stock_data(tickers, start=start_date, end=end_date):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    returns = data.pct_change().dropna()   # Convert prices to percentage returns
    return returns

# Fetching the historical data for both stocks and market index
returns = get_stock_data(STOCKS + [MARKET_INDEX])
market_returns = returns[MARKET_INDEX]  # Extract only the market index returns
returns = returns.drop(columns=[MARKET_INDEX])  # Remove the market from individual stock data

# 2. CAPM Regression (Calculate Alpha, Beta, and Idiosyncratic Variance)
betas, alphas, idiosyncratic_variance = {}, {}, {}

for stock in STOCKS:
    X = sm.add_constant(market_returns - RISK_FREE_RATE)    # Independent variable: Market Excess Return
    y = returns[stock] - RISK_FREE_RATE    # Dependent variable: Stock Excess Return
    model = sm.OLS(y, X).fit()    # Perform Ordinary Least Squares (OLS) Regression

    alphas[stock] = model.params["const"]   # Alpha: Stock's excess return independent of market
    betas[stock] = model.params[MARKET_INDEX]   # Beta: Stock's sensitivity to market movements
    idiosyncratic_variance[stock] = np.var(model.resid) # Residual variance (stock-specific risk)

# Compute expected returns
market_mean = market_returns.mean()
expected_returns = {stock: RISK_FREE_RATE + betas[stock] * (market_mean - RISK_FREE_RATE) for stock in STOCKS}

# Visualizing CAPM Regression Results with a Scatter Plot
norm = colors.Normalize(vmin=min(idiosyncratic_variance.values()), vmax=max(idiosyncratic_variance.values()))
cmap = cm.get_cmap("coolwarm")  # Color map for variance visualization

# Assign edge colors based on normalized idiosyncratic variance
edge_colors = [cmap(norm(v)) for v in idiosyncratic_variance.values()]

# Create scatter plot (Alpha vs Beta) with idiosyncratic risk as color scale
plt.figure(figsize=(12, 6))
plt.scatter(list(betas.values()), list(alphas.values()), s=100, c=list(idiosyncratic_variance.values()), cmap="coolwarm",
            edgecolors=edge_colors, linewidth=1.5, alpha=0.8)

# Add annotations for each stock
for i, stock in enumerate(STOCKS):
    plt.annotate(stock, (list(betas.values())[i], list(alphas.values())[i]), fontsize=9, xytext=(10,-4), textcoords="offset points")

# Formatting the plot
plt.xlabel("Beta (β) - Market Sensitivity")
plt.ylabel("Alpha (α) - Excess Return")
plt.title("Risk and Return Dynamics: Alpha vs Beta with Idiosyncratic Variance")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)  # Reference line at α = 0
plt.axvline(1, color="gray", linestyle="--", linewidth=0.8)  # Reference line at β = 1
plt.colorbar(label="Idiosyncratic Variance")
plt.grid(True, linestyle="--", alpha=0.6)

# 3. Portfolio Optimization
cov_matrix = np.zeros((len(STOCKS), len(STOCKS)))
beta_vals = np.array(list(betas.values()))
market_var = np.var(market_returns)    # Market variance calculation

# Compute the Covariance Matrix
for i in range(len(STOCKS)):
    for j in range(len(STOCKS)):
        if i != j:
            cov_matrix[i, j] = beta_vals[i] * beta_vals[j] * market_var    # Cross-stock covariance
    cov_matrix[i, i] = beta_vals[i] ** 2 * market_var + idiosyncratic_variance[STOCKS[i]]   # Own-stock variance

# Convert to DataFrame for better visualization
cov_df = pd.DataFrame(cov_matrix, index=STOCKS, columns=STOCKS)

# Compute the average covariance for each stock
avg_covariance = cov_df.mean(axis=1)

# Plot line chart (Stock vs. Average Covariance)
plt.figure(figsize=(10, 5))
plt.plot(STOCKS, avg_covariance, marker='o', linestyle='-', color='b')
plt.xlabel("Stocks")
plt.ylabel("Average Covariance")
plt.title("Average Covariance of Each Stock with Others")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)

# Define objective function
def portfolio_variance(weights):
    """Objective function: Minimize portfolio variance."""
    return weights.T @ cov_matrix @ weights

# Constraints
def target_return_constraint(weights, target_return):
    """Constraint: Portfolio should meet target return."""
    return np.dot(weights, list(expected_returns.values())) - target_return

def sum_weights_constraint(weights):
    """Constraint: Portfolio weights must sum to 1."""
    return np.sum(weights) - 1

# Optimize portfolio weights for different target returns
def optimize_portfolio(target_return):
    num_STOCKS = len(STOCKS)
    init_guess = np.ones(num_STOCKS) / num_STOCKS    # Start with equal weights
    bounds = [(0, 1) for _ in range(num_STOCKS)]    # No short-selling constraint
    constraints = [
        {'type': 'eq', 'fun': sum_weights_constraint},
        {'type': 'eq', 'fun': target_return_constraint, 'args': (target_return,)}
    ]
    result = sco.minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints = constraints)
    if result.success:
        return result.x
    else:
        print(f"Optimization failed for target return {target_return}: {result.message}")
        return None

target_returns = np.linspace(min(expected_returns.values()), max(expected_returns.values()), 10)
optimal_portfolios = [(optimize_portfolio(mu)) for mu in target_returns]

# Create a DataFrame for better visualization
target_returns = np.round(np.linspace(min(expected_returns.values()), max(expected_returns.values()), 10),4)
df_portfolio = pd.DataFrame(optimal_portfolios, columns=STOCKS, index=target_returns)

# Creating a heatmap with a different color tone (Blues) for better contrast
plt.figure(figsize=(12, 6))
sns.heatmap(df_portfolio, annot=True, fmt=".4f", cmap="Blues", linewidths=0.5)
plt.xlabel("Stocks")
plt.ylabel("target_returns")
plt.title("Target Returns vs Optimal Portfolio Weights")


# 4. Efficient Frontier
target_returns = np.linspace(min(expected_returns.values()), max(expected_returns.values()), 10)
efficient_frontier = [np.sqrt(portfolio_variance(optimize_portfolio(mu))) for mu in target_returns]

# Plot Efficient Frontier
plt.figure(figsize=(10, 8))
plt.plot(efficient_frontier, target_returns, marker='o', linestyle='-', label='Efficient Frontier')
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.legend()
plt.grid()
plt.show()
