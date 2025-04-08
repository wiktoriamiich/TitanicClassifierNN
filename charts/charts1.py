import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, roc_auc_score, \
    precision_score, recall_score, f1_score, log_loss, matthews_corrcoef, average_precision_score


import os
import matplotlib.pyplot as plt

def plot_training_history(history, title="Model Training History", filename_prefix="", folder="plots"):
    """
    Rysowanie wykresów strat, dokładności i MSE oraz zapisywanie ich jako pliki w określonym folderze.
    """
    # Sprawdzamy, czy folder istnieje, jeśli nie to go tworzymy
    if not os.path.exists(folder):
        os.makedirs(folder)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    # Dodanie MSE (jeśli jest dostępne w historii treningu)
    mse = history.get('mse', None)
    val_mse = history.get('val_mse', None)

    epochs = range(1, len(acc) + 1)

    # Wykres dokładności
    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    accuracy_plot_path = os.path.join(folder, f'{filename_prefix}_accuracy.png')
    plt.savefig(accuracy_plot_path)  # Zapis wykresu jako plik PNG w folderze
    plt.show()

    # Wykres strat (loss)
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    loss_plot_path = os.path.join(folder, f'{filename_prefix}_loss.png')
    plt.savefig(loss_plot_path)  # Zapis wykresu jako plik PNG w folderze
    plt.show()

    # Wykres MSE (jeśli istnieje)
    if mse is not None:
        plt.figure()
        plt.plot(epochs, mse, label='Training MSE', color='blue')
        plt.plot(epochs, val_mse, label='Validation MSE', color='red')
        plt.title('Training and Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        mse_plot_path = os.path.join(folder, f'{filename_prefix}_mse.png')
        plt.savefig(mse_plot_path)  # Zapis wykresu jako plik PNG w folderze
        plt.show()

    print(f"All plots saved in '{folder}' folder.")

def plot_additional_metrics(custom_metrics, filename_prefix="", folder="plots"):
    """Rysowanie wykresów niestandardowych metryk"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    epochs = range(1, len(custom_metrics.val_f1s) + 1)

    # F1-Score
    plt.figure()
    plt.plot(epochs, custom_metrics.val_f1s, label='F1-Score', marker='o', color='blue')
    plt.title('F1-Score - validation data')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.grid()
    plt.legend()
    f1_plot_path = os.path.join(folder, f'{filename_prefix}_f1_score.png')
    plt.savefig(f1_plot_path)  # Zapis wykresu jako plik PNG
    plt.show()

    # Precision
    plt.figure()
    plt.plot(epochs, custom_metrics.val_precisions, label='Precision', marker='o', color='orange')
    plt.title('Precision - validation data')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend()
    precision_plot_path = os.path.join(folder, f'{filename_prefix}_precision.png')
    plt.savefig(precision_plot_path)  # Zapis wykresu jako plik PNG
    plt.show()

    # ROC-AUC
    plt.figure()
    plt.plot(epochs, custom_metrics.val_rocs, label='ROC-AUC', marker='o', color='green')
    plt.title('ROC-AUC - validation data')
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC')
    plt.grid()
    plt.legend()
    roc_auc_plot_path = os.path.join(folder, f'{filename_prefix}_roc_auc.png')
    plt.savefig(roc_auc_plot_path)  # Zapis wykresu jako plik PNG
    plt.show()


def plot_confusion_matrix(y_true, y_pred, filename_prefix="", folder="plots"):
"""
Rysowanie i zapisywanie macierzy konfuzji na podstawie podanych rzeczywistych etykiet (y_true)
i przewidywanych etykiet (y_pred).

Parametry:
- y_true: rzeczywiste etykiety klas.
- y_pred: przewidywane etykiety klas.
- filename_prefix: prefiks dla nazwy pliku zapisywanej macierzy.
- folder: folder, w którym zostanie zapisany wykres macierzy konfuzji.

Działanie:
- Tworzenie folderu, jeśli nie istnieje.
- Obliczanie macierzy konfuzji.
- Tworzenie wykresu macierzy konfuzji.
- Zapisywanie wykresu w formacie PNG w podanym folderze.
"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_plot_path = os.path.join(folder, f'{filename_prefix}_confusion_matrix.png')
    plt.savefig(cm_plot_path)  # Zapis wykresu jako plik PNG
    plt.show()


def plot_misclassified_samples(y_true, y_pred, filename_prefix="", folder="plots"):
    """
Obliczanie różnych metryk oceny modelu, wyświetlanie je w konsoli i zapisywanie do pliku tekstowego.

Parametry:
- y_true: rzeczywiste etykiety klas.
- y_pred: przewidywane etykiety klas.
- y_pred_prob: przewidywane prawdopodobieństwa dla klas.
- folder: folder, w którym zostaną zapisane wyniki.
- filename: nazwa pliku, w którym zostaną zapisane wyniki.

Działanie:
- Tworzyenie folderu, jeśli nie istnieje.
- Obliczanie metryki, takie jak dokładność, F1, ROC-AUC, precyzję, czułość, stratę logarytmiczną, itp.
- Wyświetlanie metryki w konsoli.
- Zapisywanie wyników do pliku tekstowego w podanym folderze.
"""
    if not os.path.exists(folder):
        os.makedirs(folder)

    errors = np.sum(y_true != y_pred)
    correct = len(y_true) - errors

    labels = ['Correct', 'Incorrect']
    values = [correct, errors]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['green', 'red'], alpha=0.7)
    plt.title('Number of Correct and Incorrect Predictions')
    plt.ylabel('Number of Samples')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Wyświetlanie wartości nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 50,  # 5 to odstęp od słupka
                 round(yval, 0), ha='center', va='bottom', fontsize=12)

    misclassified_plot_path = os.path.join(folder, f'{filename_prefix}_misclassified_samples.png')
    plt.savefig(misclassified_plot_path)  # Zapis wykresu jako plik PNG
    plt.show()


def plot_classification_error(y_true, y_pred, title="Classification Error", filename_prefix="", folder="plots"):
    """
Tworzenie wykresów błędów klasyfikacji na podstawie różnicy między rzeczywistymi (y_true)
i przewidywanymi (y_pred) wartościami.

Parametry:
- y_true: rzeczywiste etykiety klas.
- y_pred: przewidywane etykiety klas.
- title: tytuł wykresu.
- filename_prefix: prefiks dla nazwy pliku zapisywanego wykresu.
- folder: folder, w którym zostanie zapisany wykres.

Działanie:
- Tworzenie folderu, jeśli nie istnieje.
- Obliczanie różnicy (błąd) między y_true i y_pred.
- Tworzenie wykresu błędów.
- Zapisywanie wykresu w formacie PNG w podanym folderze.
"""

    if not os.path.exists(folder):
        os.makedirs(folder)

    error = y_true - y_pred.flatten()
    mse_error = mean_squared_error(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(error)), error)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{title} (MSE={mse_error:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Error (y_true - y_pred)')
    classification_error_plot_path = os.path.join(folder, f'{filename_prefix}_classification_error.png')
    plt.savefig(classification_error_plot_path)  # Zapis wykresu jako plik PNG
    # plt.show()


def plot_weights(model, filename_prefix="", folder="plots"):
    """
Tworzenie wykresów słupkowych przedstawiających wartości wag modelu dla każdej warstwy.

Parametry:
- model: wytrenowany model Keras.
- filename_prefix: prefiks dla nazwy pliku zapisywanego wykresu.
- folder: folder, w którym zostaną zapisane wykresy.

Działanie:
- Tworzenie folder, jeśli nie istnieje.
- Iteracja przez warstwy modelu.
- Dla każdej warstwy z wagami tworzenie wykresu słupkowego przedstawiającego wartości wag.
- Zapisywanie wykresów w formacie PNG w podanym folderze.
"""

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, layer in enumerate(model.layers):
        # Sprawdzenie, czy warstwa ma wagi (nie wszystkie warstwy je mają, np. Dropout)
        if len(layer.get_weights()) > 0:
            weights, biases = layer.get_weights()  # Pobranie wag i biasów

            # Flatten wag dla wygodnego wyświetlenia na osi X
            flat_weights = weights.flatten()
            x_values = np.arange(len(flat_weights))  # Numer wagi na osi X

            # Rysowanie wykresu słupkowego dla wag
            plt.figure(figsize=(12, 6))
            plt.bar(x_values, flat_weights, color='skyblue')
            plt.title(f'Weights - {layer.name}')
            plt.xlabel('Weight number')
            plt.ylabel('Weight value')
            plt.grid(True, linestyle='--', alpha=0.7)
            weights_plot_path = os.path.join(folder, f'{filename_prefix}_weights_{layer.name}.png')
            plt.savefig(weights_plot_path)  # Zapis wykresu jako plik PNG
            # plt.show()


def plot_comparison_bar_chart(y_true, y_pred, filename_prefix="", folder="plots"):
    """
Tworzenie wykresu słupkowego porównującego rzeczywiste liczności klas (y_true) oraz liczbę
poprawnie sklasyfikowanych próbek (y_pred).

Parametry:
- y_true: rzeczywiste etykiety klas.
- y_pred: przewidywane etykiety klas.
- filename_prefix: prefiks dla nazwy pliku zapisywanego wykresu.
- folder: folder, w którym zostanie zapisany wykres.

Działanie:
- Tworzenie folderu, jeśli nie istnieje.
- Obliczanie liczby rzeczywistych etykiet klas 0 i 1 oraz poprawnych klasyfikacji.
- Tworzenie wykresu słupkowego porównujący te liczby.
- Zapisywanie wykresów w formacie PNG w podanym folderze.
"""

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Liczenie liczby elementów rzeczywistych 0 i 1
    true_0 = np.sum(y_true == 0)
    true_1 = np.sum(y_true == 1)

    # Liczenie liczby poprawnie sklasyfikowanych elementów dla 0 i 1
    correct_0 = np.sum((y_true == 0) & (y_pred == 0))
    correct_1 = np.sum((y_true == 1) & (y_pred == 1))

    # Przygotowanie danych do wykresu
    true_values = [true_0, true_1]
    correct_values = [correct_0, correct_1]

    # Tworzenie wykresu słupkowego
    ind = np.arange(2)  # Indeksy dla dwóch klas (0 i 1)
    width = 0.35  # Szerokość słupków

    fig, ax = plt.subplots(figsize=(8, 6))

    # Rysowanie słupków dla wartości rzeczywistych i poprawnie sklasyfikowanych
    rects1 = ax.bar(ind - width / 2, true_values, width, label='True Values', color='skyblue')
    rects2 = ax.bar(ind + width / 2, correct_values, width, label='Correctly Classified', color='green')

    # Dodanie etykiet, tytułu i legendy
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Comparison of True Values vs Correctly Classified Samples')
    ax.set_xticks(ind)
    ax.set_xticklabels(['Not survived', 'Survived'])
    ax.legend()

    # Dodanie etykiet liczbowych nad słupkami
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Przesunięcie etykiety
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    # Wyświetlenie wykresu
    plt.tight_layout()
    comparison_plot_path = os.path.join(folder, f'{filename_prefix}_comparison_bar_chart.png')
    plt.savefig(comparison_plot_path)  # Zapis wykresu jako plik PNG
    plt.show()


def save_all_metrics(y_true, y_pred, y_pred_prob, folder="metrics", filename="all_metrics.txt"):
    """
Obliczanie różnych metryk oceny modelu, wyświetlanie je w konsoli i zapisywanie do pliku tekstowego.

Parametry:
- y_true: rzeczywiste etykiety klas.
- y_pred: przewidywane etykiety klas.
- y_pred_prob: przewidywane prawdopodobieństwa dla klas.
- folder: folder, w którym zostaną zapisane wyniki.
- filename: nazwa pliku, w którym zostaną zapisane wyniki.

Działanie:
- Tworzyenie folderu, jeśli nie istnieje.
- Obliczanie metryki, takie jak dokładność, F1, ROC-AUC, precyzję, czułość, stratę logarytmiczną, itp.
- Wyświetlanie metryki w konsoli.
- Zapisywanie wyników do pliku tekstowego w podanym folderze.
"""

    # Sprawdzamy, czy folder istnieje, jeśli nie to go tworzymy
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Tworzymy pełną ścieżkę do pliku
    file_path = os.path.join(folder, filename)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    mse = mean_squared_error(y_true, y_pred)

    # Wyświetlanie wyników na ekranie
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"ROC-AUC: {auc_roc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Log Loss: {loss}")
    print(f"MCC: {mcc}")
    print(f"Precision-Recall AUC: {pr_auc}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Zapis do pliku
    with open(file_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"ROC-AUC: {auc_roc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Log Loss: {loss}\n")
        f.write(f"MCC: {mcc}\n")
        f.write(f"Precision-Recall AUC: {pr_auc}\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")

    print(f"All metrics saved to {filename}")


