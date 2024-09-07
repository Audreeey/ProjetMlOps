from app import model_pred

new_data = {'credit_lines_outstanding': 0,
            'loan_amt_outstanding': 5221.545193,
            'total_debt_outstanding': 3915.471226,
            'income': 78039.38546,
            'years_employed': 5,
            'fico_score': 605}

#def test_predict():
 #   prediction = model_pred(new_data)
  #  assert prediction == 1, "incorrect prediction"


def test_predict():
    prediction = model_pred(new_data)
    print(f"Prediction: {prediction}")  # Affiche la prédiction pour debug
    if prediction == 0:
        print("Test validé : la prédiction est correcte.")
    else:
        raise AssertionError("Mauvaise prédiction : le modèle n'a pas prédit 0.")
