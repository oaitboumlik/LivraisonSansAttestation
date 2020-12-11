# LivraisonSansAttestation

- src/preprocess contient le code sous un fichier python qui sert à effectuer la preprocessing des données dans le notebook `1.Data_preparation.ipynb`
- /src/model contient un fichier python contenant les fonctions qui permettent de comparer différents modèles et le notebook qui lui est associé permet d'observer les performances du modèle de Gradient Boosting au cours de l'année
- src/3.Train_model.ipynb effectue l'entrainement et l'optimisation des hyperparamètres pour le modèle retenu. Il effectue aussi une observation sur les résultats qu'obtient le modèle.

- /data/covid_data: Données covid recueillies par Oxford dans leur indicateur OxCGRT
- /data/holidays: Holidays data
- /data/final_data: Output of /src/preprocess/Data_preparation.ipynb where the prepared data is stored
- /data/raw_data : Raw data downloaded from opendata.paris.fr

Le modèle final est entrainé dans le notebook `src/3.Train_model.ipynb`.

Des tests des différents modèles sont réalisés dans les dossiers Tests et final nn.
