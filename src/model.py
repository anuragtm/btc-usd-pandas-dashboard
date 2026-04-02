from pathlib import Path
import pandas as pd
import sklearn

#Finding project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1] #resolve gives full absolute path
DATA_PATH = PROJECT_ROOT / "data" / "btcusd.csv"

print("scikit-learn imported sucessfully")
print("Project root:", PROJECT_ROOT)
print("Data path:", DATA_PATH)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True) #sort old to new(for clean row)

print("\nOriginal shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

#features
df["daily_return"] = df["Adj Close"].pct_change()
df["return_lag1"] = df["daily_return"].shift(1)
df["return_lag2"] = df["daily_return"].shift(2)

#target
df["next_day_up"] = (df["Adj Close"].shift(-1) > df["Adj Close"]).astype(int) #shift(-1) pulls tomorrow price upward into todays row, astype(T to 1, F to 0)

print("\nData with new columns:")
print(df[["Date", "Adj Close", "daily_return", "return_lag1", "return_lag2", "next_day_up"]].head(10))

#needed columns for ML
model_df = df[["Date", "Adj Close", "daily_return", "return_lag1", "return_lag2", "next_day_up"].copy()]

#missing values
model_df = model_df.dropna().reset_index(drop=True)
model_df = model_df.iloc[:-1].reset_index(drop=True)

print("\n Clean modeling dataset shape:", model_df.shape)
print(model_df.head(10))

print("\n Last 5 rows of cleaning model dataset:")
print(model_df.tail())

print("\n Target value counts:")
print(model_df["next_day_up"].value_counts())