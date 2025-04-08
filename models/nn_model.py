import tensorflow as tf
from keras import layers, models
from keras.api.optimizers import schedules
from keras.src.optimizers import Adam, RMSprop, SGD
from keras.src.regularizers import regularizers
from keras.src.saving import get_custom_objects, register_keras_serializable


# Model sieci neuronowej
def create_model(input_shape):
    """
Tworzenie podstawowego modelu sieci neuronowej z jedną ukrytą warstwą i regularyzacją Dropout.

Parametry:
- input_shape: liczba cech wejściowych.

Zwraca skompilowany model Keras do klasyfikacji binarnej.
"""
    # Tworzenie modelu sekwencyjnego
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2, name='dropout_layer_1'),  # Wyłączenie 30% neuronów
        layers.Dense(1, activation='sigmoid')  # Klasyfikacja binarna
    ])
    # Kompilacja modelu z optymalizatorem Adam, funkcją strat i metryką dokładności
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_model_smaller_a(input_shape):
    """
Tworzenie modelu z harmonogramem zmniejszania współczynnika uczenia i regularizacją L2.

Parametry:
- input_shape: liczba cech wejściowych.

Zwraca skompilowany model Keras z dodatkowymi metrykami.
"""
    # Definicja harmonogramu zmniejszania współczynnika uczenia
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizerRMS = RMSprop(learning_rate=0.001, momentum=0.9)
    optimizerSGD = SGD(learning_rate=0.001, momentum=0.9)
    optimizerAdam2 = Adam(learning_rate=lr_schedule)

    optimizerAdam = Adam(learning_rate=0.001)

    # Tworzenie modelu sekwencyjnego
    model = models.Sequential([
        layers.Dense(10, activation='relu', input_shape=(input_shape,), name='hidden_layer_1',
                     kernel_regularizer=regularizers.L2(0.001)),
        layers.Dropout(0.2, name='dropout_layer_1'),  # Wyłączenie 20% neuronów
        layers.Dense(1, activation='sigmoid', name='output_layer',
                     kernel_regularizer=regularizers.L2(0.001)),    # Klasyfikacja binarna
    ])

    # Kompilacja modelu z optymalizatorem Adam, funkcją strat i dodatkowymi metrykami
    model.compile(optimizer=optimizerAdam,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    return model

@register_keras_serializable(package="Custom")
def focal_loss(alpha=0.25, gamma=2.0):
    """
Definiowanie niestandardowej funkcji strat Focal Loss do obsługi nierównowagi klas.

Parametry:
- alpha: waga dla klasy dodatniej (domyślnie 0.25).
- gamma: parametr ogniskowania (domyślnie 2.0).

Zwraca funkcję strat Focal Loss.
"""
    def loss(y_true, y_pred):
        # Stabilizacja numeryczna
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Focal Loss dla klasy 1 i klasy 0
        focal_loss_pos = -alpha * tf.pow(1 - y_pred, gamma) * y_true * tf.math.log(y_pred)
        focal_loss_neg = -(1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true) * tf.math.log(1 - y_pred)

        # Połączenie strat
        loss = tf.reduce_mean(focal_loss_pos + focal_loss_neg)
        return loss

    return loss


def create_model_smaller_b(input_shape):
    """
Tworzenie modelu z harmonogramem zmniejszania współczynnika uczenia i Focal Loss.

Parametry:
- input_shape: liczba cech wejściowych.

Zwraca skompilowany model Keras z funkcją strat Focal Loss.
"""
    # Ustalenie harmonogramu zmniejszania współczynnika uczenia (learning rate)
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=0.001,  # Początkowa wartość współczynnika uczenia (learning rate)
        decay_steps=10000,  # Liczba kroków (iteracji lub batchów) przed każdą aktualizacją współczynnika uczenia
        decay_rate=0.9  # Współczynnik, o który będzie zmniejszany współczynnik uczenia
    )
    optimizerRMS = RMSprop(learning_rate=0.001, momentum=0.9)
    optimizerSGD = SGD(learning_rate=0.001, momentum=0.9)
    optimizerAdam2 = Adam(learning_rate=lr_schedule)

    optimizerAdam = Adam(learning_rate=0.001)

    # Definicja Focal Loss jako funcji strat
    focal_loss_fn = focal_loss(alpha=0.62, gamma=2)  # Użycie Focal Loss
    get_custom_objects().update({"focal_loss": focal_loss(alpha=0.62, gamma=2)})

    # Tworzenie modelu sekwencyjnego
    model = models.Sequential([
        layers.Dense(10, activation='relu', input_shape=(input_shape,), name='hidden_layer_1',
        kernel_regularizer = regularizers.L2(0.001)),  # L2 regularization
        layers.Dropout(0.2, name='dropout_layer_1'),  # Wyłączenie 20% neuronów
        layers.Dense(1, activation='sigmoid', name='output_layer',
                     kernel_regularizer=regularizers.L2(0.001)),    # Klasyfikacja binarna
    ])

    # Kompilacja modelu z optymalizatorem Adam, Focal Loss i dodatkowymi metrykami
    model.compile(optimizer=optimizerAdam,
                  loss=focal_loss_fn,
                  metrics=['accuracy', 'mse'])

    return model

def create_model_medium(input_shape):
    """
Tworzenie średniej wielkości modelu sieci neuronowej z Dropout i regularyzacją.

Parametry:
- input_shape: liczba cech wejściowych.

Zwraca skompilowany model Keras do klasyfikacji binarnej.
"""
    # Tworzenie modelu sekwencyjnego
    model = models.Sequential([
        layers.Dense(12, activation='relu', input_shape=(input_shape,), name='hidden_layer_1'),
        layers.Dropout(0.2, name='dropout_layer_1'),
        layers.Dense(1, activation='sigmoid', name='output_layer')  # Klasyfikacja binarna
    ])

    # Kompilacja modelu z optymalizatorem Adam, Focal Loss i dodatkowymi metrykami
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'mse'])

    return model
