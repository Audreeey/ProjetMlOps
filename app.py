from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("random_forest_model.pkl", "rb"))  # Chargez votre modèle mis à jour

def model_pred(features):
    # Convertir les caractéristiques en DataFrame avec les noms de colonnes correspondants
    feature_names = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
                     'income', 'years_employed', 'fico_score']
    test_data = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(test_data)
    return int(prediction[0])

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Extraire les caractéristiques du formulaire
            credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
            loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
            total_debt_outstanding = float(request.form["total_debt_outstanding"])
            income = float(request.form["income"])
            years_employed = int(request.form["years_employed"])
            fico_score = int(request.form["fico_score"])

            # Créer un dictionnaire de caractéristiques
            features = {
                'credit_lines_outstanding': credit_lines_outstanding,
                'loan_amt_outstanding': loan_amt_outstanding,
                'total_debt_outstanding': total_debt_outstanding,
                'income': income,
                'years_employed': years_employed,
                'fico_score': fico_score
            }

            # Prédiction du modèle
            prediction = model_pred(features)

            # Affichage du résultat
            if prediction == 0:
                return render_template(
                    "index.html",
                    prediction_text="This customer is unlikely to default."
                )
            else:
                return render_template(
                    "index.html", 
                    prediction_text="This customer is likely to default!"
                )
        except Exception as e:
            # Affiche l'erreur dans la console et renvoie un message d'erreur
            print(f"An error occurred: {e}")
            return render_template("index.html", prediction_text="An error occurred. Please try again.")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
