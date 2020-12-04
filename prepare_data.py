import pandas as pd

def preprocess(df: pd.DataFrame):
    """
    Enlève toutes les colonnes sauf la date, le débit horaire et le taux d'occupation
    et formatte les dates en type datetime
    pour ensuite en extraire année, jour, mois, heure et jour de la semaine
    Réordonne les données temporellement
    """
    temp = df[["Débit horaire", "Taux d'occupation"]].copy()
    temp["Date et heure de comptage"] = pd.to_datetime(df["Date et heure de comptage"], utc=True)
    temp = temp.sort_values("Date et heure de comptage")
    temp = temp.set_index("Date et heure de comptage")
    temp["datetime"] = temp.index
    temp["year"] = temp.index.year
    temp["month"] = temp.index.month
    temp["day"] = temp.index.day
    temp["hour"] = temp.index.hour
    # Récupère les jours de la semaine : 0 -> lundi, 6 -> dimanche
    temp["dayofweek"] = temp.index.dayofweek
    return temp