from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


horizon = 6
initial = 300 # en jours
period = 3

def eval_pred(data, model, initial=initial*24, horizon=horizon*24, period=period*24):
    """
    Genere un dataframe des erreurs de prédiction connaissant le passé
    (ie) evaluation de la prédiction en tenant compte de la structure temporelle
    data : données entières sur lesquels on va entrainer et testeer
    model : modèle à tester, il doit avoir les deux méthodes suivantes :
        fit et predict
    initial : nombre d'heures minimales d'entrainement
    horizon : horizon de prédiction en heures
    period : période de répétition de la prédiction en heures
    """
    datetime_data = pd.to_datetime(data["datetime"])
    X = data.drop(columns=["Débit horaire", "Taux d'occupation", "datetime", "Date"])
    N = len(X)
    initial_idx = np.arange(initial, N-horizon, period)
    result_debit = pd.DataFrame()
    result_occupation = pd.DataFrame()
    for idx in tqdm(initial_idx):
        X_train, X_test = X[:idx], X[idx:idx+horizon]
        debit_train, debit_test = data["Débit horaire"][:idx], data["Débit horaire"][idx:idx+horizon]
        occupation_train, occupation_test = data["Taux d'occupation"][:idx], data["Taux d'occupation"][idx:idx+horizon]
        model.fit(X_train, debit_train)
        pred_debit = model.predict(X_test)
        model.fit(X_train, occupation_train)
        pred_occupation = model.predict(X_test)
        result_debit = result_debit.append(pd.DataFrame({
            "ds":datetime_data[idx:idx+horizon],
            "yhat": pred_debit,
            "y": debit_test,
            "cutoff": datetime_data[idx-1]
        }))
        result_occupation = result_occupation.append(pd.DataFrame({
            "ds":datetime_data[idx:idx+horizon],
            "yhat": pred_occupation,
            "y": occupation_test,
            "cutoff": datetime_data[idx-1]
        }))
    return result_debit, result_occupation

def compute_error(g):
    r2 = r2_score(g["y"], g["yhat"])
    mae = mean_absolute_error(g["y"], g["yhat"])
    rmse = mean_squared_error(g["y"], g["yhat"], squared=False)
    nmse = np.mean((g["y"] - g["yhat"])**2/g["y"]**2)
    return pd.Series({"r2": r2, "mae": mae, "rmse":rmse, "nmse": nmse})

def error_prediction(prediction_data):
    temp = prediction_data.copy()
    temp["delay"] = temp["ds"] - temp["cutoff"]
    plt.title("RMSE prediction by prediction")
    temp.groupby("cutoff").apply(compute_error)["rmse"].plot()
    plt.show()
    return temp.groupby("cutoff").apply(compute_error)["rmse"]
