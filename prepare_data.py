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
    temp[["Débit horaire", "Taux d'occupation"]] = temp[["Débit horaire", "Taux d'occupation"]].interpolate("time")
    return temp

def get_covid_data():
    """
    Récupère les données covid pour la France récoltées par Oxford
    Ces données apporte des informations sur les restrictions gouvernementales
    Une explication des données covid est disponible dans le fichier data/doc-oxford.pdf
    """
    oxford_data = pd.read_csv("data/OxCGRT_latest.csv", sep=";")
    france_data = oxford_data[oxford_data["CountryName"] == "France"]
    france_data = france_data[['Date', 'C1_School closing',
        'C2_Workplace closing', 'C3_Cancel public events',
        'C4_Restrictions on gatherings', 'C5_Close public transport',
        'C6_Stay at home requirements',
        'C7_Restrictions on internal movement', 'StringencyLegacyIndexForDisplay',
        'ContainmentHealthIndexForDisplay',]]
    france_data["Date"] = pd.to_datetime(france_data["Date"].astype(str).apply(lambda date: f"{date[:4]}-{date[4:6]}-{date[6:]}"))
    france_data.index = france_data["Date"]
    france_data = france_data.interpolate("time")
    france_data = france_data.reset_index(drop=True)
    return france_data

def add_covid_data(df: pd.DataFrame, covid_data_france:pd.DataFrame):
    """
    df est la base de données à laquelle on veut ajouter les données covid
    df doit contenir une colonne datetime de type datetime
    !Attention! : pour éviter les problèmes lors de la jointure la colonne index de la 
    base de données est supprimée
    """
    # Préparation de la base de données pour le join
    temp = df.copy()
    temp["Date"] = pd.to_datetime(temp.datetime.dt.date)
    temp = temp.reset_index(drop=True)

    # Join
    # Comme les données manquantes sont uniquement celles avant 2020
    # la valeurs des indicateurs associées peut être 0
    joined = pd.merge(temp, covid_data_france, how="left", on="Date").fillna(0)
    return joined
