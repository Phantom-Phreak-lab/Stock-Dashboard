

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from statsmodels.tsa.arima.model import ARIMA


# ================= REGRESSION =================
def run_ml_pipeline(df):
    df = df.copy()
    df = df.dropna()

    if len(df) < 50:
        return None, {}, {}, [], None

    features = ["Open", "High", "Low", "Volume", "MA20", "MA50", "Volatility"]

    X = df[features]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100, verbosity=0)
    }
    

    results = {}
    predictions_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        score = r2_score(y_test, preds)
        results[name] = score
        predictions_dict[name] = preds

    arima_score, arima_prediction, arima_preds, arima_y_test = run_arima_model(df)
    if arima_score is not None:
        results["ARIMA"] = arima_score
        predictions_dict["ARIMA"] = arima_preds

    best_model_name = max(results, key=results.get)
    if best_model_name == "ARIMA":
        prediction = arima_prediction
        y_test = arima_y_test
    else:
        best_model = models[best_model_name]
        best_model.fit(X, y)

        last_row = X.iloc[-1:].copy()
        prediction = best_model.predict(last_row)[0]

    return best_model_name, results, predictions_dict, y_test, prediction


# ================= CLASSIFICATION =================
def run_classification_pipeline(df):
    df = df.copy()
    df = df.dropna()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    features = ["Open", "High", "Low", "Volume", "MA20", "MA50", "Volatility", "RSI", "MACD"]

    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }


    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc)

    best_model_name = max(results, key=lambda x: results[x][1])
    best_model, best_acc = results[best_model_name]

    last_row = X.iloc[-1:].copy()

    prob = best_model.predict_proba(last_row)[0]
    prediction = best_model.predict(last_row)[0]
    confidence = max(prob)
    

    return best_model_name, best_acc, prediction, confidence
#==========================ARIMA MODEL===========================
def run_arima_model(df):
    df = df.copy()
    df = df.dropna()

    if len(df) < 50:
        return None, None, None, None

    try:
        series = df["Close"]

        # Train/test split (last 20% test)
        split = int(len(series) * 0.8)
        train, test = series[:split], series[split:]

        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()

        preds = model_fit.forecast(steps=len(test))

        score = r2_score(test, preds)

        # Next prediction
        next_pred = model_fit.forecast(steps=1).iloc[0]

        return score, next_pred, preds, test

    except Exception as e:
        print("ARIMA error:", e)
        return None, None, None, None