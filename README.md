# BTC-USD Pandas Dashboard

A small pandas project that downloads BTC-USD pair daily price data and generates:
- summary metrics (return, volatility, max drawdown)
- monthly returns table
- charts: price with SMA20/SMA50, drawdown and rolling 20 day volaitlity

## Project Structure
- 'src/': Python scripts
- 'data/': downloaded BTC-USD CSV
- 'outputs/': generated CSV reports and charts

## Setup
```bash
pip3 install -r requirements.txt