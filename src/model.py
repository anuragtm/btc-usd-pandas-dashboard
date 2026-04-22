from pathlib import Path
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1] #resolve gives full absolute path
DATA_PATH = PROJECT_ROOT / "data" / "btcusd.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUT_DIR / "ml_report.txt"

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
#trendFeatures
df["sma20"] = df["Close"].rolling(20).mean()    #short-term
df["sma50"] = df["Close"].rolling(50).mean()  #lt
df["sma_gap_pct"] = ((df["sma20"] / df["sma50"]) - 1) * 100 
#Volatility
df["vol20"] = df["daily_return"].rolling(20).std()
df["vol20_pct"] = df["vol20"] * 100
#drawdownFeat
df["peak"] = df["Adj Close"].cummax()
df["drawdown"] = (df["Adj Close"] / df["peak"]) - 1
df["drawdown_pct"] = df["drawdown"] * 100

#target
df["next_day_up"] = (df["Adj Close"].shift(-1) > df["Adj Close"]).astype(int) #shift(-1) pulls tomorrow price upward into todays row, astype(T to 1, F to 0)

feature_cols = ["daily_return", "return_lag1", "return_lag2", "sma20", "sma50", "sma_gap_pct", "vol20_pct", "drawdown_pct",]

#build clean modeling dataset
model_df = df[["Date"] + feature_cols + ["next_day_up"]].copy()
model_df = model_df.dropna().reset_index(drop=True)
model_df = model_df.iloc[:-1].reset_index(drop=True)

print("\n Clean modeling dataset shape:", model_df.shape)

#inputFeatues x =inputFeat, y=target
x = model_df[feature_cols]
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

#buildin&train modelPipeline
log_model = Pipeline([("scaler", StandardScaler()),
("logistic_regression", LogisticRegression(max_iter=1000))])

log_model.fit(X_train, y_train)#modelTrain
log_pred = log_model.predict(X_test) #predict on test set
log_accuracy = accuracy_score(y_test, log_pred) #actual answers vs predicted ans
print(f"\n Logistic Regression Accuracy: {log_accuracy:.4f}")

results_df = pd.DataFrame({
    "Date": test_dates.values,
    "Actual": y_test.values,
    "Predicted": log_pred})
print("\n First 10 predictions vs actual:")
print(results_df.head(10))
print("\n Actual value counts in test set:")
print(y_test.value_counts())
print("\nPredicted value countds:")
print(pd.Series(log_pred).value_counts())
#above for what real test set distribution VS what the model predicted

baseline_up_accuracy =  (y_test == 1).mean()
baseline_down_accuracy = (y_test == 0).mean()
print("\n Baseline accuracy if always predicting 1/up:")
print(f"{baseline_up_accuracy:.4f}")
print("\nBaseline accuracy if always predicting 0/down:")
print(f"{baseline_down_accuracy:.4f}")

log_cm = confusion_matrix(y_test, log_pred)
print("\nLogistic Regression Confusion Matrix:")
print("[[TN, FP],") # 0(down)0,1
print(" [FN, TP]]") #1(up)0,1
print(log_cm)


log_report = classification_report(y_test, log_pred, target_names=["Not Up / Down", "Up"])

print("\nLogistic Regression Classification Report:")
print(log_report)
#random forest model

rf_model = RandomForestClassifier(n_estimators = 200, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)#predict on test set
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("\nRandom Forest predicted value counts:")
print(pd.Series(rf_pred).value_counts())

rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print("[[TN, FP],")
print(" [FN, TP]]")
print(rf_cm)

rf_report = classification_report(y_test, rf_pred, target_names=["Not Up / Down", "Up"])
print("\nRandom Forest Classification Report:")
print(rf_report)

report_lines = []

report_lines.append("BTC-USD Next-Day Direction ML Report")
report_lines.append("=" * 50)
report_lines.append("")
report_lines.append("Target:")
report_lines.append("next_day_up = 1 if tomorrow's Adj Close is higher than today's Adj Close, else 0")
report_lines.append("")

report_lines.append("Features used:")
for col in feature_cols:
    report_lines.append(f"- {col}")

report_lines.append("")
report_lines.append(f"Clean modeling dataset shape: {model_df.shape}")
report_lines.append(f"Train shape: {X_train.shape}")
report_lines.append(f"Test shape: {X_test.shape}")
report_lines.append("")

report_lines.append(f"Train date range: {train_dates.iloc[0]} to {train_dates.iloc[-1]}")
report_lines.append(f"Test date range: {test_dates.iloc[0]} to {test_dates.iloc[-1]}")
report_lines.append("")

report_lines.append("Baseline Results")
report_lines.append("-" * 50)
report_lines.append(f"Baseline accuracy if always predicting 1/up: {baseline_up_accuracy:.4f}")
report_lines.append(f"Baseline accuracy if always predicting 0/down: {baseline_down_accuracy:.4f}")
report_lines.append("")

report_lines.append("Logistic Regression Results")
report_lines.append("-" * 50)
report_lines.append(f"Accuracy: {log_accuracy:.4f}")
report_lines.append("")
report_lines.append("Predicted value counts:")
report_lines.append(str(pd.Series(log_pred).value_counts()))
report_lines.append("")
report_lines.append("Confusion Matrix:")
report_lines.append("[[TN, FP],")
report_lines.append(" [FN, TP]]")
report_lines.append(str(log_cm))
report_lines.append("")
report_lines.append("Classification Report:")
report_lines.append(log_report)
report_lines.append("")

report_lines.append("Random Forest Results")
report_lines.append("-" * 50)
report_lines.append(f"Accuracy: {rf_accuracy:.4f}")
report_lines.append("")
report_lines.append("Predicted value counts:")
report_lines.append(str(pd.Series(rf_pred).value_counts()))
report_lines.append("")
report_lines.append("Confusion Matrix:")
report_lines.append("[[TN, FP],")
report_lines.append(" [FN, TP]]")
report_lines.append(str(rf_cm))
report_lines.append("")
report_lines.append("Classification Report:")
report_lines.append(rf_report)
report_lines.append("")

report_lines.append("Short Conclusion")
report_lines.append("-" * 50)
report_lines.append(
    "Both models showed weak predictive performance for BTC next-day direction. "
    "Logistic Regression and Random Forest were close to the simple baseline, "
    "which suggests that the current return, trend, volatility, and drawdown features "
    "do not provide a strong signal for reliable next-day direction prediction."
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

print(f"\nSaved ML report to: {REPORT_PATH}")