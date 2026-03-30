from pathlib import Path
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "btcusd.csv"

def main():
    ticker = "BTC-USD"

    df = yf.download(ticker, period="max", interval="1d", auto_adjust=False)

    if df.empty:
        raise RuntimeError("No data downloaded. Check the internet connection and try again.")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index() #make date a normal column(not index); means turns date index into noraml date column

    df["Ticker"] = ticker

    #saving to csv
    OUT_PATH.parent.mkdir(exist_ok=True) #if it aint found any folder, create new parent one
    df.to_csv(OUT_PATH, index=False) #prevents pandas from writing extra index info into csv

    print(f"Saved {len(df)} rows to {OUT_PATH}") #len to count how many rows were downloa..

if __name__ == "__main__":
    main()
