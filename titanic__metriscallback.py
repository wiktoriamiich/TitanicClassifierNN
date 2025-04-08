import keras
from sklearn.metrics import f1_score, roc_auc_score, precision_score
import numpy as np

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        """"
        Inicjalizacja klasy MetricsCallback, która jest rozszerzeniem rozszerzeniem Keras Callback.
        Służy do obliczania wskaźników walidacyjnych (F1, precyzji oraz ROC-AUC) na końcu każdej epoki podczas trenowania modelu.

        Atrybuty:
        - validation_data: dane walidacyjne (cechy i etykiety).
        - val_f1s: lista przechowująca wartości F1 z każdej epoki.
        - val_precisions: lista przechowująca wartości precyzji z każdej epoki.
        - val_rocs: lista przechowująca wartości ROC-AUC z każdej epoki.

        Metody:
        - on_epoch_end: oblicza wskaźniki na końcu każdej epoki i zapisuje wyniki.
        """

    # Inicjalizacja callbacku z danymi walidacyjnymi
        self.validation_data = validation_data
        self.val_f1s = []
        self.val_precisions = []
        self.val_rocs = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Funkcja wykonywana na końcu każdej epoki.
        Oblicza wskaźniki walidacyjne i zapisuje ich wartości.

        Parametry:
        - epoch: numer bieżącej epoki.
        - logs: słownik z logami treningu (opcjonalny).
        """
        # Rozdzielenie danych walidacyjnych na wejście i etykiety i progowanie predykcji modelu
        val_x, val_y = self.validation_data
        val_pred = (self.model.predict(val_x) > 0.5).astype(int)

        # Obliczanie wskaźników
        val_f1 = f1_score(val_y, val_pred)
        val_precision = precision_score(val_y, val_pred)
        val_roc = roc_auc_score(val_y, self.model.predict(val_x))

        # Dodawanie wyników do list
        self.val_f1s.append(val_f1)
        self.val_precisions.append(val_precision)
        self.val_rocs.append(val_roc)

        # Wyświetlanie wyników w konsoli
        print(f"Epoch {epoch + 1}: val_f1={val_f1:.4f}, val_precision={val_precision:.4f}, val_roc={val_roc:.4f}")


import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, precision_score

class CustomMetricsCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir="./logs"):
        """
        Inicjalizacja klasy CustomMetricsCallback - rozszerzenie klasy MetricsCallback o możliwość zapisywania metryk walidacyjnych do TensorBoard.

        Atrybuty:
        - validation_data: dane walidacyjne (cechy i etykiety).
        - val_f1s: lista przechowująca wartości F1 z każdej epoki.
        - val_precisions: lista przechowująca wartości precyzji z każdej epoki.
        - val_rocs: lista przechowująca wartości ROC-AUC z każdej epoki.
        - log_dir: ścieżka do katalogu logów TensorBoard.
        - file_writer: obiekt do zapisu logów TensorBoard.

        Metody:
        - on_epoch_end: oblicza wskaźniki na końcu każdej epoki, zapisuje je do TensorBoard i wyświetla w konsoli.
        """
        # Inicjalizacja callbacku z danymi walidacyjnymi i katalogiem logów TensorBoard
        self.validation_data = validation_data
        self.val_f1s = []
        self.val_precisions = []
        self.val_rocs = []
        self.log_dir = log_dir                                        # Ścieżka do katalogu logów
        self.file_writer = tf.summary.create_file_writer(log_dir)     # Tworzenie obiektu do zapisu logów

    def on_epoch_end(self, epoch, logs=None):
        """
        Funkcja wykonywana na końcu każdej epoki.
        Oblicza wskaźniki walidacyjne, zapisuje ich wartości do TensorBoard i wyświetla w konsoli.

        Parametry:
        - epoch: numer bieżącej epoki.
        - logs: słownik z logami treningu (opcjonalny).
        """
        
        # Rozdzielenie danych walidacyjnych na wejście i etykiety, predykcje i progowanie
        val_x, val_y = self.validation_data
        val_pred_proba = self.model.predict(val_x)
        val_pred = (val_pred_proba > 0.5).astype(int)

        # Obliczanie wskaźników
        val_f1 = f1_score(val_y, val_pred)
        val_precision = precision_score(val_y, val_pred)
        val_roc = roc_auc_score(val_y, self.model.predict(val_x))

        self.val_f1s.append(val_f1)
        self.val_precisions.append(val_precision)
        self.val_rocs.append(val_roc)

        # Zapisanie metryk do TensorBoard
        with self.file_writer.as_default():
            tf.summary.scalar("val_f1", val_f1, step=epoch)
            tf.summary.scalar("val_precision", val_precision, step=epoch)
            tf.summary.scalar("val_roc_auc", val_roc, step=epoch)

        print(f"Epoch {epoch+1}: val_f1={val_f1:.4f}, val_precision={val_precision:.4f}, val_roc_auc={val_roc:.4f}")
