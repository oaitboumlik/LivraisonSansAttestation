import csv
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from datetime import timedelta
import os.path



def read_and_save_csv_columns(read_path: str, write_path: str, delimiter: str, *col_names: str):
    df = pd.read_csv(read_path, delimiter=delimiter)
    df = pd.DataFrame(df, columns=col_names)
    print(len(df) * 3 -2)
    df["Date"] = df["Date"].map(lambda x: pd.to_datetime(x[:-6], format="%Y-%m-%dT%H:%M:%S"))
    df.sort_values("Date", inplace=True, ignore_index=True)
    df.iloc[30277, 1] = df.iloc[30276, 1]
    df.iloc[30277, 2] = df.iloc[30276, 2]

    new_df = pd.DataFrame(columns=col_names)
    # new_df["Date"] = pd.date_range(df["Date"][0], df["Date"][len(df) - 1], (len(df) - 1) * 3 + 1, "H")
    for index, row in tqdm(df.iterrows()):
        i = int(index)
        date0, temp0, weather0 = [row[n] for n in col_names]
        new_df.loc[i * 3] = [date0, round(temp0, 2), round(weather0, 2)]

        if i == len(df) - 1:
            break
        try:
            date1, temp1, weather1 = [df.loc[i + 1][n] for n in col_names]
            delta_time, delta_temp = (date1 - date0) / 3, (temp1 - temp0) / 3
            date_i1, date_i2 = date0 + delta_time, date1 - delta_time
            temp_i1, temp_i2 = temp0 + delta_temp, temp1 - delta_temp

            new_df.loc[i * 3 + 1] = [date_i1, round(temp_i1, 2), round(weather0, 2)]
            new_df.loc[i * 3 + 2] = [date_i2, round(temp_i2, 2), round(weather1, 2)]
        except:
            print(df.loc[i])
            continue

    new_df.columns = ["Date", "Température", "Temps présent"]
    print(len(new_df))
    print(new_df.loc[0])
    print(new_df.loc[:5])
    new_df.to_csv(write_path, sep=delimiter, index=False)
    print("write csv finish")

def trim_traffic(path, to_path):
    df = pd.read_csv(path, sep=";")
    date_col = "Date"
    df[date_col] = df[date_col].map(lambda x: pd.to_datetime(x[:-6], format="%Y-%m-%dT%H:%M:%S"))

    df = pd.DataFrame(df,
                      columns=["Date", "Débit horaire", "Taux d'occupation"])
    df.sort_values("Date", inplace=True, ignore_index=True)
    for index, row in tqdm(df.iterrows()):
        i = int(index)
        if i == 0:
            continue
        date, c1, c2 = [row[n] for n in ["Date", "Débit horaire", "Taux d'occupation"]]
        if pd.isna(c1):
            df.iloc[i, 1] = df.iloc[i - 1, 1]
        if pd.isna(c2):
            df.iloc[i, 2] = df.iloc[i - 1, 2]

    df["Débit horaire1"] = df["Débit horaire"]
    df["Taux d'occupation1"] = df["Taux d'occupation"]
    df.to_csv(to_path, sep=";", index=False)

def trafic_into_np(path):
    df = pd.read_csv(path, sep=";")

    data_seq = df.iloc[:, 1:].values
    data_seq = data_seq.reshape((data_seq.shape[0], 2, int(data_seq.shape[1]) // 2))
    print(np.where(np.isnan(data_seq)))
    # print(data_seq[101])
    # print(data_seq[416])
    # print(data_seq[566])
    return data_seq


def concat_meteo_traffic(meteo_path, traff_path, write_path, delimiter):
    df_meteo = pd.read_csv(meteo_path, delimiter=delimiter)
    df_traff = pd.read_csv(traff_path, delimiter=delimiter)
    date_col = "Date"
    df_meteo[date_col] = df_meteo[date_col].map(lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S"))
    df_traff[date_col] = df_traff[date_col].map(lambda x: pd.to_datetime(x[:-6], format="%Y-%m-%dT%H:%M:%S"))

    df_merge = pd.merge(df_meteo, df_traff, on=date_col, how='inner')

    print(df_merge.loc[:5])
    df_merge.to_csv(write_path, sep=delimiter, index=False)

def open_npz(path):
    # np.savez_compressed(path)
    b = np.load(path)['data']
    print(b.shape)
    for idx in range(b.shape[0]):
        print(b[idx])
        break

def plot_trafic(path):
    seq = trafic_into_np(path)
    plt.plot(seq[:24 * 14, 1, 0])
    plt.show()



if __name__ == '__main__':
    # read_and_save_csv_columns('../data/meteo/meteo_all.csv', '../data/meteo/meteo_copie.csv', ";", "Date", "Température (°C)",
    #                           "Temps présent1")

    # trim_traffic("../data/traffic/Champs-Elysées.csv", "../data/traffic/ChmpElyse_clean.csv")
    # print(trafic_into_np("../data/traffic/ChmpElyse_clean.csv"))
    plot_trafic("../data/traffic/ChmpElyse_clean.csv")
    # concat_meteo_traffic("../data/meteo/meteo_copie.csv", "../data/traffic/Champs-Elysées.csv",
    #                      "../data/merge_meteo_Champs_Elysees.csv", ";")

    # open_npz("../biblio/基于时空注意力机制的图卷积神经网络/pems04.npz")

