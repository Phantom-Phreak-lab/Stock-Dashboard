from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from xgboost import XGBRegressor


def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=100, verbosity=0)
    }

    results = {}
    predictions_dict = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        score = r2_score(y_test, predictions)

        results[name] = score
        predictions_dict[name] = predictions

    return results, predictions_dict