import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("data/processed/maritime_accidents.csv")

os.makedirs("prediction", exist_ok=True)
os.makedirs("visualization/prediction", exist_ok=True)

TEST_YEARS = [2023, 2024, 2025]
START_TRAIN_YEAR = 2014

def validate_model():

    evolution = (df.groupby("annee").size().reset_index(name="nb_accidents"))

    train = evolution[(evolution["annee"] >= START_TRAIN_YEAR)
        & (~evolution["annee"].isin(TEST_YEARS))]

    test = evolution[evolution["annee"].isin(TEST_YEARS)]

    X_train = train[["annee"]]
    y_train = train["nb_accidents"]

    X_test = test[["annee"]]
    y_test = test["nb_accidents"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    r2_train = r2_score(y_train,model.predict(X_train))

    print(f"Slope (accidents/year): {model.coef_[0]:.2f}")
    print(f"R² on training data: {r2_train:.3f}")
    print(f"MAE on 2023-2025: {mae:.1f} accidents")
    print(f"RMSE on 2023-2025: {rmse:.1f} accidents")

    for year, actual, predicted in zip(test["annee"],y_test,test_predictions):
        gap_pct = (predicted - actual) / actual * 100

        print(
            f"{year}: "
            f"actual={actual} | "
            f"predicted={predicted:.0f} | "
            f"gap={gap_pct:+.1f}%"
        )

    plt.figure(figsize=(10, 5))
    plt.plot(evolution["annee"],evolution["nb_accidents"],marker="o",label="Actual data")

    regression_predictions = model.predict(train[["annee"]])

    plt.plot(
        train["annee"],
        regression_predictions,
        linestyle="--",
        label="Linear regression"
    )

    # Prédictions 2023-2025
    plt.plot(
        test["annee"],
        test_predictions,
        marker="x",
        linestyle="--",
        markersize=10,
        label="Predictions (2023-2025)"
    )

    plt.title(
        "Linear regression validation "
        "(training: 2014-2022)")
    plt.xlabel("Year")
    plt.ylabel("Number of accidents")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualization/prediction/validation_regression.png",dpi=150)
    plt.close()

    return model, mae, rmse


if __name__ == "__main__":
    model, mae, rmse = validate_model()