from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "btcusd.csv"

OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def main():
 
    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"]) #converts date or stringdate to real data type
    df = df.sort_values("Date").reset_index(drop=True) #after sorting, reset_index resets them to 0,1,2,3..cleanly

    num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") #csv stores numbers as text & can have weird characters, so using coerce 

    df["daily_return"] = df["Adj Close"].pct_change() #for pct, =today - yesterday close / yesterday's c

    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    df["vol20"] = df["daily_return"].rolling(20).std() 
    df["vol20_pct"] = df["vol20"]*100

    df["peak"] = df["Adj Close"].cummax() #cummax = running max
    df["drawdown"] = (df["Adj Close"]) / df["peak"] - 1 
    df["drawdown_pct"] = df["drawdown"] * 100

    summary = pd.DataFrame([{"start_date": df["Date"].min(), "end_date": df["Date"].max(), "rows":len(df),
                             "start_price": df["Adj Close"].iloc[0], "end_price": df["Adj Close"].iloc[-1],
                             "total_return": (df["Adj Close"].iloc[-1] / df["Adj Close"].iloc[0]) - 1, #(end / start - 1)
                             "avg_daily_return": df["daily_return"].mean(),
                             "vol20_latest": df["vol20"].iloc[-1],
                             "max_drawdown": df["drawdown"].min()

                            }])
    summary["total_return_pct"] = summary["total_return"] * 100
    summary["avg_daily_return_pct"] = summary["avg_daily_return"] *100
    summary["vol20_latest_pct"] = summary["vol20_latest"] * 100
    summary["max_drawdown_pct"] = summary["max_drawdown"] * 100

    summary_path = OUT_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    monthly = df.set_index("Date")["Adj Close"].resample("ME").last().pct_change()
    monthly = monthly.reset_index().rename(columns={"Adj Close": "monthly_return"})
    monthly["monthly_return_pct"] = monthly["monthly_return"] * 100

    monthly_path = OUT_DIR / "monthly_return.csv"
    monthly.to_csv(monthly_path, index=False)
    print(f"Saved {monthly_path}")

    print("Rows, Columns:", df.shape)
    print("Date Range:", df["Date"].min(), "to", df["Date"].max())

    print("\n New columns added:", ["daily_return", "sma20", "sma50"])
    print("\n Last 5 rows (Date, Close, daily_return, sma20, sma50): ")
    print(df[["Date", "Adj Close", "daily_return", "sma20", "sma50", "vol20", "drawdown"]].tail())

    # worst =  df.loc[df["drawdown"].idxmin(), ["Date", "Adj Close", "peak", "drawdown"]]
    # print("\nWorst drawdown point")
    # print(worst)
    # print("\nWorst drawdown percent:", f"{worst['drawdown'] * 100:.2f}%")

    worst_i = df["drawdown"].idxmin()
    peak_i = df.loc[:worst_i, "Adj Close"].idxmax()

    worst_row = df.loc[worst_i, ["Date", "Adj Close", "drawdown"]]
    peak_row = df.loc[peak_i, ["Date", "Adj Close", "drawdown"]]

    print("\n Worst drawdown day:")
    print(worst_row)
    print("\nPeak day (before that drawdown):")
    print(peak_row)
    print("\nWorst drawdown percent:", f"{df.loc[worst_i, 'drawdown']*100:.2f}%")

    plt.figure() #starting new empty chart for sma 20 & 50 comparison

    plt.plot(df["Date"], df["Close"], label="Close")
    plt.plot(df["Date"], df["sma20"], label="SMA20")
    plt.plot(df["Date"], df["sma50"], label="SMA50")
    plt.title("BTC-USD Close price with SMA20 & SMA50")
    plt.xlabel("Date")
    plt.ylabel("Price USD")
    plt.legend()

    chart_path = OUT_DIR / "price_sma.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight") #saving img File
    plt.close()
    print(f"Saved{chart_path}")
 

    #drawdownnnn
    plt.figure()
    plt.plot(df["Date"], df["drawdown_pct"], label="Drawdown (%)")

    #for max drawdown point
    worst_i = df["drawdown"].idxmin()
    worst_date = df.loc[worst_i, "Date"]
    worst_dd_pct = df.loc[worst_i, "drawdown_pct"]

    plt.scatter([worst_date], [worst_dd_pct], label="Max drawdown", zorder=5)

    plt.annotate(
        f"{worst_dd_pct:.2f}%",
        (worst_date, worst_dd_pct),
        textcoords="offset points",
        xytext=(10, -10)
    )

    plt.title("BTC-USD Drawdown (from running peak)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (in %)")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    dd_path = OUT_DIR / "drawdown.png"
    plt.savefig(dd_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {dd_path}")

    #volatility 20 day
    plt.figure()
    plt.plot(df["Date"], df["vol20_pct"], label="20D Volatility (%)")

    vol_i = df["vol20"].idxmax()
    vol_date = df.loc[vol_i, "Date"]
    vol_val_pct = df.loc[vol_i, "vol20_pct"]
    
    plt.scatter([vol_date], [vol_val_pct], label = "Max volatility", zorder=5)
    plt.annotate(
        f"{vol_val_pct:.2f}%",
        (vol_date, vol_val_pct),
        textcoords="offset points",
        xytext=(10, -10)
    )

    plt.title("BTC-USD Rolling 20 Day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility in (%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    vol_path = OUT_DIR / "Volatility_20d.png"
    plt.savefig(vol_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {vol_path}")


if __name__ == "__main__":
    main()