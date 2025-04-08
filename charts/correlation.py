import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
data_path = 'data/train.csv'
data = pd.read_csv(data_path)

# Przetwarzanie danych
# Zamiana kolumn tekstowych na wartości numeryczne
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})  # Zakładając, że wartości są C, Q, S

# Usunięcie niepotrzebnych kolumn
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Wypełnianie brakujących wartości
data = data.fillna(data.median())

# Obliczenie macierzy korelacji
correlation_matrix = data.corr()

# Wizualizacja heatmapy
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Macierz korelacji cech - Titanic Dataset")
plt.show()
