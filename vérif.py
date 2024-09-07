import pickle

# Chargez votre modèle à partir du fichier pickle
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Vérifiez les caractéristiques utilisées pour entraîner le modèle
if hasattr(model, "feature_names_in_"):
    print("Colonnes utilisées pour entraîner le modèle :")
    print(model.feature_names_in_)
else:
    print("L'attribut 'feature_names_in_' n'est pas disponible pour ce modèle.")
