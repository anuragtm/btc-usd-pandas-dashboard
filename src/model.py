from pathlib import Path
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

#build clean modeling dataset
model_df = df[["Date", "daily_return", "return_lag1", "return_lag2", "next_day_up"]].copy()
model_df = model_df.dropna().reset_index(drop=True)
model_df = model_df.iloc[:-1].reset_index(drop=True)

print("\n Clean modeling dataset shape:", model_df.shape)

#inputFeatues
x = model_df[["daily_return", "return_lag1", "return_lag2"]]
y = model_df["next_day_up"]

#train&test split

split_index = int(len(model_df) * 0.8)

X_train = x.iloc[:split_index]
X_test = x.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

train_dates = model_df["Date"].iloc[:split_index]
test_dates = model_df["Date"].iloc[split_index:]

print("\n Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain data range:")
print(train_dates.iloc[0], "to", train_dates.iloc[-1])

print("\nTest data range:")
print(test_dates.iloc[0], "to", test_dates.iloc[-1])

model = LogisticRegression(max_iter=1000) #Build model
model.fit(X_train,y_train) #train 
y_pred = model.predict(X_test) #predict on test set

accuracy = accuracy_score(y_test, y_pred) #actual answers vs predicted ans
print(f"\n Logistic Regression Accuracy: {accuracy:.4f}")

results_df = pd.DataFrame({
    "Date": test_dates.values,
    "Actual": y_test.values,
    "Predicted": y_pred

})

print("\n First 10 predictions vs actual:")
print(results_df.head(10))