import numpy as np
from keras.src.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from titanic__metriscallback import CustomMetricsCallback
from wykresy import plot_training_history, plot_weights, plot_classification_error, plot_misclassified_samples, \
    plot_confusion_matrix, plot_comparison_bar_chart, plot_additional_metrics, save_all_metrics
from models.titanic_model import TitanicModel
import model_keras
from sklearn.metrics import accuracy_score, classification_report
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

processor = TitanicModel(train_path='data/train.csv', test_path='data/test.csv')

def model1(processor):
    processor.load_data1()        # Wczytanie danych
    processor.preprocess_data1()  # Wstępne przetwarzanie danych
    processor.prepare_data1()  # Przygotowanie danych

    x_train1 = processor.x1
    y_train1 = processor.y1

    # Trening i ocena modelu na pierwszym zbiorze
    model1 = model_keras.create_model(input_shape=x_train1.shape[1])
    model1.summary()

    """ Trening modelu 
    - epochs - liczba pełnych przejść przez cały zbiór treningowy
    - batch_size - oznacza liczbę próbek, które model przetwarza przed aktualizacją wag
    - validation_split=0.2 - 20% danych treningowych będzie używane jako dane walidacyjne
    - verbose=1 - wyniki treningu będą wyświetlane w konsoli
    """
    history1 = model1.fit(
        x_train1, y_train1,
        epochs=20, batch_size=32,
        validation_split=0.2, verbose=1
    )

    # Ewaluacja modelu
    y_pred1 = (model1.predict(x_train1) > 0.5).astype(int)
    print("Accuracy for Dataset 1:", accuracy_score(y_train1, y_pred1))
    print(classification_report(y_train1, y_pred1))

    model1.save('models/save_models/model1.h5')

    # Dla zbioru 1
    with open('models/save_models/scaler1.pkl', 'wb') as f:
        pickle.dump(processor.scaler1, f)
    # Wywołanie funkcji plotującej historie
    plot_training_history(history1, title="Model 1 (Dataset 1)")

def model2a(processor):
    """ MODEL 2a - mniejszy
        - wsp uczenia
        - momentum
        - dropout
        - wczesne kończenie
        - regularyzacja
    """
    processor.load_data2()  # Wczytanie danych
    processor.preprocess_data2()  # Wstępne przetwarzanie danych
    processor.prepare_data2()  # Przygotowanie danych
    processor.show_headers(2)

    x_train2 = processor.x2
    y_train2 = processor.y2

    # Podział na dane treningowe i walidacyjne (jeśli nie zrobiono wcześniej)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train2,  # Zbiór cech wejściowych (np. 'Pclass', 'Age', 'Fare', etc.)
        y_train2,  # Zbiór etykiet (np. 0 lub 1, czy pasażer przeżył)
        test_size=0.2,  # 20% danych zostaje przeznaczone na zbiór walidacyjny
        random_state=42,  # Ziarno dla generatora liczb pseudolosowych, aby wyniki były powtarzalne
        stratify=y_train2  # Zachowanie proporcji klas w podziale danych (np. 80% klasy 0 i 20% klasy 1)
    )

    # Stworzenie modelu
    model2 = model_keras.create_model_smaller_a(input_shape=x_train.shape[1])
    model2.summary()

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # Callback do metryk niestandardowych
    custom_metrics = CustomMetricsCallback(validation_data=(x_val, y_val), log_dir="./logs")

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   min_delta=0.001,  # Min. zmiana błędu, by kontynuować
                                   restore_best_weights=True)
    """ Trening modelu 
    - epochs - liczba pełnych przejść przez cały zbiór treningowy
    - batch_size - oznacza liczbę próbek, które model przetwarza przed aktualizacją wag
    - validation_split=0.2 - 20% danych treningowych będzie używane jako dane walidacyjne
    - verbose=1 - wyniki treningu będą wyświetlane w konsoli
    """
    history2 = model2.fit(
        x_train, y_train,
        epochs=300, batch_size=20,
        verbose=1,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, custom_metrics],
    )

    # Ewaluacja modelu
    y_pred2 = (model2.predict(x_train2) > 0.5).astype(int)
    # Zapisanie wszystkich metryk
    save_all_metrics(y_train2, y_pred2, model2.predict(x_train2), folder="pomiary", filename="model2a_all_metrics.txt")

    # Konwersja do jednowymiarowej tablicy, jeśli potrzebna
    y_pred2 = y_pred2.flatten()  # Zamiana (891, 1) na (891,)
    y_train2 = y_train2.values.ravel()  # Zamiana Pandas Series na NumPy array

    # Porównanie wyników predykcji z etykietami rzeczywistymi
    errors = np.where(y_pred2 != y_train2)[0]
    print("Błędnie sklasyfikowane próbki:", errors)

    model2.save('models/save_models/model2a.h5')

    # Dla zbioru 2
    with open('models/save_models/scaler2a.pkl', 'wb') as f:
        pickle.dump(processor.scaler2, f)

    # Wywołanie funkcji plotującej historie
    plot_training_history(history2.history, folder="pomiary", filename_prefix="model2a")
    plot_additional_metrics(custom_metrics, folder="pomiary", filename_prefix="model2a")
    plot_comparison_bar_chart(y_train2, y_pred2, folder="pomiary", filename_prefix="model2a")
    plot_misclassified_samples(y_train2, y_pred2, folder="pomiary", filename_prefix="model2a")
    plot_confusion_matrix(y_train2, y_pred2, folder="pomiary", filename_prefix="model2a")
    plot_classification_error(y_train2, y_pred2, folder="pomiary", title="Classification Error (Threshold 0.5)", filename_prefix="model2a")
    plot_weights(model=model2, folder="pomiary", filename_prefix="model2a")

def model2b(processor):
    """ MODEL 2b - mniejszy
        - wsp uczenia
        - momentum
        - dropout
        - wczesne kończenie
        - równoważenie różnicy przetrwań i nieprzetrwań
        - regularizacja
    """
    processor.load_data2()  # Wczytanie danych
    processor.preprocess_data2()  # Wstępne przetwarzanie danych
    processor.prepare_data2()  # Przygotowanie danych
    processor.show_headers(2)

    x_train2 = processor.x2
    y_train2 = processor.y2

    # Podział na dane treningowe i walidacyjne (jeśli nie zrobiono wcześniej)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train2,  # Zbiór cech wejściowych (np. 'Pclass', 'Age', 'Fare', etc.)
        y_train2,  # Zbiór etykiet (np. 0 lub 1, czy pasażer przeżył)
        test_size=0.2,  # 20% danych zostaje przeznaczone na zbiór walidacyjny
        random_state=42,  # Ziarno dla generatora liczb pseudolosowych, aby wyniki były powtarzalne
        stratify=y_train2  # Zachowanie proporcji klas w podziale danych (np. 80% klasy 0 i 20% klasy 1)
    )

    # Trening i ocena modelu na drugim zbiorze
    model2 = model_keras.create_model_smaller_b(input_shape=x_train.shape[1])
    model2.summary()

    # Upewnij się, że y_train jest np.array
    y_train = np.array(y_train)

    # Callback do metryk niestandardowych
    custom_metrics = CustomMetricsCallback(validation_data=(x_val, y_val), log_dir="./logs")

    # Obliczanie wag klas na podstawie danych treningowych
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    # Konwertowanie na słownik (format, którego używa Keras)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   min_delta=0.001,  # Minimalna zmiana błędu, by kontynuować
                                   restore_best_weights=True)

    """ Trening modelu 
    - epochs - liczba pełnych przejść przez cały zbiór treningowy
    - batch_size - oznacza liczbę próbek, które model przetwarza przed aktualizacją wag
    - validation_split=0.2 - 20% danych treningowych będzie używane jako dane walidacyjne
    - verbose=1 - wyniki treningu będą wyświetlane w konsoli
    """
    history2 = model2.fit(
        x_train, y_train,
        epochs=300, batch_size=20,
        verbose=1,
        validation_data=(x_val, y_val),  # Zamiast validation_split
        callbacks=[early_stopping, custom_metrics],
        # class_weight=class_weight_dict  # Uwzględnia wagę klas
    )

    # Ewaluacja modelu
    y_pred2 = (model2.predict(x_train2) > 0.5).astype(int)
    # Zapisanie wszystkich metryk
    save_all_metrics(y_train2, y_pred2, model2.predict(x_train2), folder="pomiary", filename="model2b_all_metrics.txt")

    # Konwersja do jednowymiarowej tablicy, jeśli potrzebna
    y_pred2 = y_pred2.flatten()  # Zamiana (891, 1) na (891,)
    y_train2 = y_train2.values.ravel()  # Zamiana Pandas Series na NumPy array

    # Porównanie wyników predykcji z etykietami rzeczywistymi
    errors = np.where(y_pred2 != y_train2)[0]
    print("Błędnie sklasyfikowane próbki:", errors)

    model2.save('models/save_models/model2b.keras')
    model2.save('models/save_models/model2b.h5')

    # Dla zbioru 2
    with open('models/save_models/scaler2b.pkl', 'wb') as f:
        pickle.dump(processor.scaler2, f)

    # Wywołanie funkcji plotującej historie
    plot_training_history(history2.history, folder="pomiary", filename_prefix="model2b")
    plot_additional_metrics(custom_metrics, folder="pomiary", filename_prefix="model2b")
    plot_comparison_bar_chart(y_train2, y_pred2, folder="pomiary", filename_prefix="model2b")
    plot_misclassified_samples(y_train2, y_pred2, folder="pomiary", filename_prefix="model2b")
    plot_confusion_matrix(y_train2, y_pred2, folder="pomiary", filename_prefix="model2b")
    plot_classification_error(y_train2, y_pred2, folder="pomiary", title="Classification Error (Threshold 0.5)", filename_prefix="model2b")
    plot_weights(model=model2, folder="pomiary", filename_prefix="model2b")


"""
    WYBÓR MODELU
"""
model2b(processor=processor)

exit(0)

def model3(processor):
    """ MODEL 3 - średni """
    processor.load_data3()  # Wczytanie danych
    processor.preprocess_data3()  # Wstępne przetwarzanie danych
    processor.prepare_data3()  # Przygotowanie danych
    processor.show_headers(3)

    x_train3 = processor.x3
    y_train3 = processor.y3

    # Trening i ocena modelu na drugim zbiorze
    model3 = model_keras.create_model_medium(input_shape=x_train3.shape[1])
    model3.summary()

    """ Trening modelu 
    - epochs - liczba pełnych przejść przez cały zbiór treningowy
    - batch_size - oznacza liczbę próbek, które model przetwarza przed aktualizacją wag
    - validation_split=0.2 - 20% danych treningowych będzie używane jako dane walidacyjne
    - verbose=1 - wyniki treningu będą wyświetlane w konsoli
    """
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   min_delta=0.001,  # Minimalna zmiana błędu, by kontynuować
                                   restore_best_weights=True)

    history3 = model3.fit(
        x_train3, y_train3,
        epochs=300, batch_size=20,
        validation_split=0.2, verbose=1,
        callbacks=[early_stopping]
    )

    # Ewaluacja modelu
    y_pred3 = (model3.predict(x_train3) > 0.5).astype(int)
    print("Accuracy for Dataset 3:", accuracy_score(y_train3, y_pred3))
    print(classification_report(y_train3, y_pred3))

    # Konwersja do jednowymiarowej tablicy, jeśli potrzebna
    y_pred3 = y_pred3.flatten()  # Zamiana (891, 1) na (891,)
    y_train3 = y_train3.values.ravel()  # Zamiana Pandas Series na NumPy array

    # Porównanie wyników predykcji z etykietami rzeczywistymi
    errors = np.where(y_pred3 != y_train3)[0]
    print("Błędnie sklasyfikowane próbki:", errors)

    model3.save('models/save_models/model3.h5')

    # Dla zbioru 3
    with open('models/saved_models/scaler3.pkl', 'wb') as f:
        pickle.dump(processor.scaler3, f)

    # Wywołanie funkcji plotującej historie
    plot_training_history(history3.history)
    plot_comparison_bar_chart(y_train3, y_pred3)
    plot_misclassified_samples(y_train3, y_pred3)
    plot_confusion_matrix(y_train3, y_pred3)
    plot_classification_error(y_train3, y_pred3, title="Classification Error (Threshold 0.5)")

    plot_weights(model=model3)



# Sprawdź dane i skonwertuj na DataFrame, jeśli to ndarray
processor.load_data2()  # Wczytanie danych
processor.preprocess_data2()  # Wstępne przetwarzanie danych
processor.prepare_data2()  # Przygotowanie danych
data = processor.x2  # lub processor.x1

if isinstance(data, np.ndarray):
    # Zakładamy, że kolumny są opisane w processor.feature_names
    data = pd.DataFrame(data, columns=processor.feature_names)

# Oblicz macierz korelacji
correlation_matrix = data.corr()

# Ustawienia wykresu
plt.figure(figsize=(10, 8))  # Rozmiar wykresu
sns.heatmap(
    correlation_matrix,
    annot=True,           # Dodaje wartości liczbowe na heatmapie
    fmt=".2f",            # Format liczb z 2 miejscami po przecinku
    cmap="coolwarm",      # Schemat kolorów (od niebieskiego do czerwonego)
    cbar=True             # Dodaj pasek skali
)

plt.title("Macierz korelacji cech wejściowych")  # Tytuł wykresu
plt.show()