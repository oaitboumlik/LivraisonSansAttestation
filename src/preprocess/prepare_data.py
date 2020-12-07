import pandas as pd
import os
import numpy as np


def preprocess(df: pd.DataFrame):
    """
    Enlève toutes les colonnes sauf la date, le débit horaire et le taux d'occupation
    et formatte les dates en type datetime
    pour ensuite en extraire année, jour, mois, heure et jour de la semaine
    Réordonne les données temporellement
    Resampling pour compléter les données absentes
    Interpolation temporelle des données
    """
    temp = df[["Débit horaire", "Taux d'occupation"]].copy()
    temp["Date et heure de comptage"] = pd.to_datetime(df["Date et heure de comptage"], utc=True)
    temp = temp.sort_values("Date et heure de comptage")
    temp = temp.set_index("Date et heure de comptage")
    temp = temp.resample("1H").first()
    temp["datetime"] = temp.index
    temp["year"] = temp.index.year
    temp["month"] = temp.index.month
    temp["day"] = temp.index.day
    temp["hour"] = temp.index.hour
    # Récupère les jours de la semaine : 0 -> lundi, 6 -> dimanche
    temp["dayofweek"] = temp.index.dayofweek
    temp[["Débit horaire", "Taux d'occupation"]] = temp[["Débit horaire", "Taux d'occupation"]].interpolate("time")
    return temp

def get_covid_data(path):
    """
    Récupère les données covid pour la France récoltées par Oxford
    Ces données apporte des informations sur les restrictions gouvernementales
    Une explication des données covid est disponible dans le fichier data/doc-oxford.pdf
    """
    oxford_data = pd.read_csv(os.path.join(path, "OxCGRT_latest.csv"),
                    sep=";")
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

def get_holidays_data(path):
    """Importer la donnée des holidays"""
    data_vacance_scolaire = pd.read_csv(os.path.join(path, 'fr-en-calendrier-scolaire.csv'),
                                        parse_dates=True, sep=';')
    data_vacance_scolaire = data_vacance_scolaire.loc[data_vacance_scolaire['location'] == 'Paris'].reset_index(drop=True)
    data_vacance_scolaire['start_date'] = data_vacance_scolaire['start_date'].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))
    data_vacance_scolaire['end_date'] = data_vacance_scolaire['end_date'].apply(lambda x: pd.to_datetime(x, format="%Y-%m-%d"))

    data_jours_feriers = pd.read_csv(os.path.join(path, 'jours_feries_metropole.csv'),
                                     parse_dates=True, sep=',')
    data_jours_feriers = data_jours_feriers.loc[data_jours_feriers['zone'] == 'Métropole'].reset_index(drop=True)
    data_jours_feriers['date'] = data_jours_feriers['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    return data_vacance_scolaire, data_jours_feriers

def add_jours_feries(df, df_ferie):
    """Joindre les jours feriers avec notre dataset"""
    joined = pd.merge(df, df_ferie[['date', 'nom_jour_ferie']],
                    left_on='Date', right_on='date', how='left').drop(['date'], axis=1)
    joined['jour_ferie'] = joined['nom_jour_ferie'].map(lambda x: 0 if pd.isnull(x)  else 1)
    return joined

def add_school_holidays(df):
    """Joindre les vacances scolaires"""
    sd_noel = pd.Timestamp('2019-12-21')
    ed_noel = pd.Timestamp('2020-01-06')

    sd_hiver = pd.Timestamp('2020-02-08')
    ed_hiver = pd.Timestamp('2020-02-24')

    sd_ete = pd.Timestamp('2020-07-04')
    ed_ete = pd.Timestamp('2020-09-01')

    sd_printemps = pd.Timestamp('2020-04-04')
    ed_printemps = pd.Timestamp('2020-04-20')

    sd_ascension = pd.Timestamp('2020-05-20')
    ed_ascension = pd.Timestamp('2020-05-25')

    sd_toussaint = pd.Timestamp('2020-10-17')
    ed_toussaint = pd.Timestamp('2020-11-02')

    def get_name_vacation(x):
        if (x >= sd_noel and x < ed_noel):
            return 'noel'
        elif (x >= sd_hiver and x < ed_hiver):
            return 'hiver'
        elif (x >= sd_ete and x < ed_ete):
            return 'été'
        elif (x >= sd_printemps and x < ed_printemps):
            return 'printemps'
        elif (x >= sd_ascension and x < ed_ascension):
            return 'ascension'
        elif (x >= sd_toussaint and x < ed_toussaint):
            return 'toussaint'
        else:
            return np.nan

    df['nom_vacance_scolaire'] = df['Date'].map(get_name_vacation)
    df['vacance_scolaire'] = df['nom_vacance_scolaire'].map(lambda x: 0 if pd.isnull(x) else 1)
    return df

