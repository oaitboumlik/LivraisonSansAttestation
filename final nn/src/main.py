from src.data_preprocessing import *
from src.run_astgcn import run_model
from src.get_model import apply_model, apply_model_real

# Champs-Elysee
# trim_traffic("Champs-Elysee")
data_file = "../data/traffic/Champs-Elysees_clean.csv"
# train model
run_model(data_file, "debit")
# test model


# Convention
# clean data
# data_file = trim_traffic("Convention")

# train model
# run_model(data_file, "debit")
# run_model(data_file, "occupancy")

# test model
debit_model_path = "../bin/model_debit_%s.params" % "207.39"
# occupation_model_path = "../bin/model_occupancy_%s.params" % "rmse here"
# apply_model(debit_model_path, data_file, "debit")
# apply_model(occupation_model_path, data_file, "occupancy")

# get result
# real_case = trim_traffic("REAL_CASE_DATA")
# debit_convention = apply_model_real(debit_model_path, data_file)
# print(debit_convention)
# occupation_convention = apply_model_real(occupation_model_path, real_case)

# Sts-peres
# clean data
# train model
# test model

