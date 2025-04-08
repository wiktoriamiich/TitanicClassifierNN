import pickle
import numpy as np
from keras.models import load_model
from keras.src.saving import get_custom_objects

from models.nn_model import focal_loss
from models.titanic_model import TitanicModel

def get_user_input():
    """
    Pobieranie danych od użytkownika i zwracanie ich jako słownik.
    """
    print("Wprowadź dane pasażera:")
    data = {
        'Pclass': int(input("Klasa (1, 2, 3): ")),
        'Sex': input("Płeć (male/female): "),
        'Age': float(input("Wiek: ")),
        'SibSp': int(input("Liczba rodzeństwa/małżonka na pokładzie: ")),
        'Parch': int(input("Liczba rodziców/dzieci na pokładzie: ")),
        'Fare': float(input("Cena biletu: ")),
        'Embarked': input("Port zaokrętowania (C/Q/S): ").upper()
    }
    return data


def preprocess_input(data, scaler):
    """
    Przetwarzanie danych wprowadzonych przez użytkownika na format zgodny z modelem.

    :param data: Dane wejściowe w formacie słownika.
    :param scaler: Załadowany scaler do normalizacji danych.
    :param model_type: Typ modelu (1 dla modelu z prefiksami biletów, 2 dla uproszczonego modelu).
    :return: Przekształcone dane jako numpy array.
    """
    # Konwersja płci i portu zaokrętowania
    data['Sex'] = 0 if data['Sex'].lower() == 'male' else 1
    embarked = {'C': 0, 'Q': 1, 'S': 2}
    data['Embarked'] = embarked.get(data['Embarked'], 2)  # Domyślnie 'S' jeśli brak danych

    # Budowanie odpowiedniego formatu wejściowego
    features = [
        data['Pclass'], data['Sex'], data['Age'], data['SibSp'], data['Parch'], data['Fare'],
        1 if data['Embarked'] == 0 else 0,  # Embarked_C
        1 if data['Embarked'] == 1 else 0   # Embarked_Q
    ]
    
    # Skalowanie cech
    features_scaled = scaler.transform([features])
    return features_scaled


def predict_survival():
    """
    Główna funkcja przewidywania.
    """
    # Wybieranie model
    print("Wybierz model (1 lub 2):")
    model_choice = int(input("1 - Model bez równoważenia klas, 2 - Model z równoważeniem klas: "))
    model_file = 'model2a.h5' if model_choice == 1 else 'model2b.keras'
    scaler_file = 'scaler2b.pkl' if model_choice == 1 else 'scaler2b.pkl'

    # Ładowanie modelu i scalera
    if model_choice==1:
        model = load_model(model_file)
    else:
        model = load_model(model_file, custom_objects={"focal_loss": focal_loss(0.62, 2)})

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    # Pobieranie danych od użytkownika
    user_data = get_user_input()

    # Przetwarzanie danych
    processed_data = preprocess_input(user_data, scaler)

    # Predykcja
    prediction = model.predict(processed_data)
    survival_chance = prediction[0][0]  # Wartość prawdopodobieństwa

    # Wyświetlanie wyniku
    print(f"Szansa na przeżycie: {survival_chance:.2%}")
    if survival_chance > 0.5:
        print("Przewidywanie: Przeżyjesz!")
    else:
        print("Przewidywanie: Niestety, nie przeżyjesz.")


if __name__ == "__main__":
    # Rejestrowanie niestandardowej funkcji straty
    get_custom_objects().update({"focal_loss": focal_loss(alpha=0.62, gamma=2)})

    while True:
        predict_survival()
