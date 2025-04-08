import pandas as pd
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


class TitanicModel:
    def __init__(self, train_path, test_path):
        """
        Inicjalizacja klasy i wczytanie danych.

        :param train_path: Ścieżka do pliku CSV z danymi treningowymi.
        :param test_path: Ścieżka do pliku CSV z danymi testowymi.

        Publiczne atrybuty klasy:
        - train_data1, test_data1: Przechowują dane surowe i wstępnie przetworzone z pierwszego zestawu podejścia.
        - train_data2, test_data2: Przechowują dane surowe i wstępnie przetworzone z drugiego zestawu podejścia.
        - x1, y1: Przechowują cechy i etykiety dla pierwszego zestawu podejścia.
        - x2, y2: Przechowują cechy i etykiety dla drugiego zestawu podejścia.
        """
        self.train_path = train_path
        self.test_path = test_path

        # Dane dla podejścia 1 (bez 'cabin', z prefiksami biletów)
        self.train_data1 = None
        self.test_data1 = None
        self.x1 = None
        self.y1 = None

        # Dane dla podejścia 2 (bez 'cabin', bez prefiksów biletów i kolumny Ticket)
        self.train_data2 = None
        self.test_data2 = None
        self.x2 = None
        self.y2 = None

        # Dane dla podejścia 3 (bez 'cabin', bez prefiksów biletów i kolumny Ticket z dodatkowymi cechami isAlone, FamilySize)
        self.train_data3 = None
        self.test_data3 = None
        self.x3 = None
        self.y3 = None

        self.feature_names = None


    def load_data1(self):
        # Wczytywanie danych z plików CSV do zestawu podejścia 1.
        self.train_data1 = pd.read_csv(self.train_path)
        self.test_data1 = pd.read_csv(self.test_path)
        print("Dane 1 zostały wczytane.")
        print(self.train_data1.head())

    def preprocess_data1(self):
        """
        Przetwarzanie danych dla podejścia 1.

        Operacje:
        - Wypełnianie brakujących danych w kolumnach Age i Fare.
        - Usuwanie kolumny Cabin.
        - Tworzenie prefiksów biletów i ich kodowanie One-Hot.
        - Dodawanie wielkości grupy biletowej.
        - Zamiana płci i kodowanie kolumny Embarked.
        """
        if self.train_data1 is None or self.test_data1 is None:
            raise ValueError("Najpierw należy wczytać dane za pomocą load_data1().")

        # Wypełnianie brakujących danych
        self.train_data1['Age'] = self.train_data1['Age'].fillna(self.train_data1['Age'].median())
        self.test_data1['Age'] = self.test_data1['Age'].fillna(self.test_data1['Age'].median())

        self.train_data1['Fare'] = self.train_data1['Fare'].fillna(self.train_data1['Fare'].median())
        self.test_data1['Fare'] = self.test_data1['Fare'].fillna(self.test_data1['Fare'].median())

        # Usunięcie Cabin
        self.train_data1.drop('Cabin', axis=1, inplace=True)
        self.test_data1.drop('Cabin', axis=1, inplace=True)

        # Prefiksy biletów
        self.train_data1['Ticket_Prefix'] = self.train_data1['Ticket'].apply(
            lambda x: x.split()[0] if len(x.split()) > 1 else 'NoPrefix')
        self.test_data1['Ticket_Prefix'] = self.test_data1['Ticket'].apply(
            lambda x: x.split()[0] if len(x.split()) > 1 else 'NoPrefix')

        # One-Hot Encoding dla prefiksów biletów
        self.train_data1 = pd.get_dummies(self.train_data1, columns=['Ticket_Prefix'], drop_first=True)
        self.test_data1 = pd.get_dummies(self.test_data1, columns=['Ticket_Prefix'], drop_first=True)

        # Wielkość grupy biletowej
        self.train_data1['Ticket_GroupSize'] = self.train_data1.groupby('Ticket')['Ticket'].transform('count')
        self.test_data1['Ticket_GroupSize'] = self.test_data1.groupby('Ticket')['Ticket'].transform('count')

        # Obsługa brakujących wartości dla Embarked
        self.train_data1['Embarked'] = self.train_data1['Embarked'].fillna(self.train_data1['Embarked'].mode()[0])
        self.test_data1['Embarked'] = self.test_data1['Embarked'].fillna(self.test_data1['Embarked'].mode()[0])

        # Mapowanie płci
        self.train_data1['Sex'] = self.train_data1['Sex'].map({'male': 0, 'female': 1})
        self.test_data1['Sex'] = self.test_data1['Sex'].map({'male': 0, 'female': 1})

        # One-Hot Encoding Embarked
        self.train_data1 = pd.get_dummies(self.train_data1, columns=['Embarked'], drop_first=True)
        self.test_data1 = pd.get_dummies(self.test_data1, columns=['Embarked'], drop_first=True)

        print("Dane 1 zostały przetworzone.")

    def prepare_data1(self):
        """
        Przygotowanie danych do trenowania dla podejścia 1.
        - Rozdzielanie cechy (X1) i etykiety (y1).
        - Skalowanie cech za pomocą StandardScaler.
        """
        # Oddzielenie cechy i etykiety
        self.y1 = self.train_data1['Survived']
        self.x1 = self.train_data1.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)

        # Skalowanie danych
        self.scaler1 = StandardScaler()
        self.x1 = self.scaler1.fit_transform(self.x1)

    def load_data2(self):
        # Wczytywanie danych z plików CSV do zestawu podejścia 2.
        self.train_data2 = pd.read_csv(self.train_path)
        self.test_data2 = pd.read_csv(self.test_path)
        print("Dane 2 zostały wczytane.")

    def preprocess_data2(self):
        """
        Przetwarzanie dane dla podejścia 2.

        Operacje:
        - Wypełnianie brakujących danych.
        - Usunięcie Cabin i Ticket.
        - Zamiana płci i kodowanie Embarked.
        """
        # Wypełnianie braków
        self.train_data2['Age'] = self.train_data2['Age'].fillna(self.train_data2['Age'].median())
        self.test_data2['Age'] = self.test_data2['Age'].fillna(self.test_data2['Age'].median())

        # Usunięcie niepotrzebnych kolumn
        self.train_data2.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
        self.test_data2.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

        # Mapowanie płci
        self.train_data2['Sex'] = self.train_data2['Sex'].map({'male': 0, 'female': 1})
        self.test_data2['Sex'] = self.test_data2['Sex'].map({'male': 0, 'female': 1})

        # One-Hot Encoding dla Embarked
        self.train_data2 = pd.get_dummies(self.train_data2, columns=['Embarked'], drop_first=True)
        self.test_data2 = pd.get_dummies(self.test_data2, columns=['Embarked'], drop_first=True)

    def prepare_data2(self):
        """
        Przygotowanie danych do trenowania dla podejścia 2.
        - Rozdzielanie cechy (X2) i etykiety (y2).
        """
        
        # Oddzielenie cech
        self.y2 = self.train_data2['Survived']
        self.x2 = self.train_data2.drop(['Survived', 'PassengerId', 'Name'], axis=1)
        self.feature_names = self.x2.columns.tolist()  # Pobieramy nazwy kolumn z x2

        # Pokazujemy zakresy po skalowaniu
        print("\nZakresy przed skalowaniem:")
        print(np.min(self.x2, axis=0))  # Minimalne wartości po skalowaniu
        print(np.max(self.x2, axis=0))  # Maksymalne wartości po skalowaniu

        # Skalowanie danych
        self.scaler2 = StandardScaler()
        self.x2 = self.scaler2.fit_transform(self.x2)

        # Pokazujemy zakresy po skalowaniu
        print("\nZakresy po skalowaniu:")
        print(np.min(self.x2, axis=0))  # Minimalne wartości po skalowaniu
        print(np.max(self.x2, axis=0))  # Maksymalne wartości po skalowaniu

        # Zapisanie nazw cech

        # self.scaler2 = MinMaxScaler()
        # self.x2 = self.scaler2.fit_transform(self.x2)

        # self.scaler2 = RobustScaler()
        # self.x2 = self.scaler2.fit_transform(self.x2)

        # self.scaler2 = Normalizer()
        # self.x2 = self.scaler2.fit_transform(self.x2)


    def load_data3(self):
        # Wczytywanie danych z plików CSV do zestawu podejścia 3.
        self.train_data3 = pd.read_csv(self.train_path)
        self.test_data3 = pd.read_csv(self.test_path)
        print("Dane 3 zostały wczytane.")

    def preprocess_data3(self):
        """
        Przetwarzanie danych dla podejścia 3.

        Operacje:
        - Wypełnianie brakujących danych.
        - Dodanie nowych cech na podstawie SibSp i Parch.
        - Usunięcie Cabin.
        - Zamiana płci i kodowanie Embarked.
        """
        
        if self.train_data3 is None or self.test_data3 is None:
            raise ValueError("Najpierw należy wczytać dane za pomocą load_data3().")

        # Wypełnianie braków
        self.train_data3['Age'] = self.train_data3['Age'].fillna(self.train_data3['Age'].median())
        self.test_data3['Age'] = self.test_data3['Age'].fillna(self.test_data3['Age'].median())

        # Dodanie nowych cech na podstawie SibSp i Parch
        self.train_data3['FamilySize'] = self.train_data3['SibSp'] + self.train_data3['Parch'] + 1  # Liczba członków rodziny
        self.test_data3['FamilySize'] = self.test_data3['SibSp'] + self.test_data3['Parch'] + 1

        self.train_data3['IsAlone'] = (self.train_data3['FamilySize'] == 1).astype(int)  # Status samotności
        self.test_data3['IsAlone'] = (self.test_data3['FamilySize'] == 1).astype(int)

        # Usunięcie niepotrzebnych kolumn
        self.train_data3.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
        self.test_data3.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

        # Mapowanie płci
        self.train_data3['Sex'] = self.train_data3['Sex'].map({'male': 0, 'female': 1})
        self.test_data3['Sex'] = self.test_data3['Sex'].map({'male': 0, 'female': 1})

        # One-Hot Encoding dla Embarked
        self.train_data3 = pd.get_dummies(self.train_data3, columns=['Embarked'], drop_first=True)
        self.test_data3 = pd.get_dummies(self.test_data3, columns=['Embarked'], drop_first=True)

        print("Dane 3 zostały przetworzone.")

    def prepare_data3(self):
        """
        Przygotowanie danych do trenowania dla podejścia 3.
        - Rozdzielanie cechy (X3) i etykiety (y3).
        - Skalowanie cech za pomocą StandardScaler.
        """
        # Oddzielenie cechy i etykiety
        self.y3 = self.train_data3['Survived']
        self.x3 = self.train_data3.drop(['Survived', 'PassengerId', 'Name'], axis=1)

        # Skalowanie danych
        self.scaler3 = StandardScaler()
        self.x3 = self.scaler3.fit_transform(self.x3)

        print("Dane 3 zostały przygotowane.")

    def show_headers(self, model_type):
        """
        Wyświetla nagłówki (nazwy kolumn) używane przez model po przetwarzaniu danych.

        :param model_type: Typ modelu ('1', '2' lub '3'), dla którego można zobaczyć nagłówki.
        """
        if str(model_type) == '1' and self.x1 is not None:
            print("Nagłówki dla modelu 1:")
            print(self.train_data1.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1).columns.tolist())
        elif str(model_type) == '2' and self.x2 is not None:
            print("Nagłówki dla modelu 2:")
            print(self.train_data2.drop(['Survived', 'PassengerId', 'Name'], axis=1).columns.tolist())
        elif str(model_type) == '3' and self.x3 is not None:
            print("Nagłówki dla modelu 3:")
            print(self.train_data3.drop(['Survived', 'PassengerId', 'Name'], axis=1).columns.tolist())
        else:
            print(f"Nie można znaleźć danych dla modelu {model_type}. Upewnij się, że dane zostały przetworzone.")


