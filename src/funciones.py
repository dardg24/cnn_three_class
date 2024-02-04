import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

def adjusted_scheduler(epoch, lr):
    """
    Ajusta la tasa de aprendizaje basada en el número de época.

    Este programador disminuye la tasa de aprendizaje exponencialmente entre las épocas 10 y 14
    y la mantiene constante a partir de entonces.

    Argumentos:
    - epoch: Un entero, el número actual de la época.
    - lr: Un flotante, la tasa de aprendizaje actual.

    Retorna:
    - Un flotante, la tasa de aprendizaje ajustada.
    """
    if epoch < 10:
        return lr  # Mantener la tasa de aprendizaje sin cambios para las primeras 10 épocas
    elif epoch < 14:
        return lr * np.exp(-0.1)  # Disminuir exponencialmente la tasa de aprendizaje
    else:
        return 6.2500e-05  # Mantener la tasa de aprendizaje constante después de la época 14

def plot_training_history(history):
    """
    Grafica la precisión y la pérdida de entrenamiento y validación para cada época.

    Argumentos:
    - history: Un objeto History de TensorFlow obtenido del método `fit` de un modelo.

    Retorna:
    Ninguno. Muestra gráficos de matplotlib de precisión/pérdida.
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    rango_epocas = range(len(acc))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
    plt.plot(rango_epocas, val_acc, label='Precisión Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión de Entrenamiento y Validación')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rango_epocas, loss, label='Pérdida Entrenamiento')
    plt.plot(rango_epocas, val_loss, label='Pérdida Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.legend()

    plt.show()

def print_epoch_logs(history):
    """
    Imprime la pérdida y precisión de los conjuntos de entrenamiento y validación en cada época.

    Argumentos:
    - history: Un objeto History de TensorFlow de una sesión de entrenamiento de un modelo.

    Retorna:
    Ninguno. Imprime la pérdida y precisión en la consola.
    """
    for i in range(len(history.history['loss'])):
        print(f"Época {i+1}, Pérdida: {history.history['loss'][i]}, Precisión: {history.history['accuracy'][i]}")

    for i in range(len(history.history['val_loss'])):
        print(f"Época {i+1}, Pérdida de Validación: {history.history['val_loss'][i]}, Precisión de Validación: {history.history['val_accuracy'][i]}")


def visualize_filters(model):
    """
    Visualiza el primer filtro de cada capa convolucional del modelo.

    Argumentos:
    - model: Una instancia de tf.keras.Model.

    Retorna:
    Ninguno. Muestra los filtros de las capas convolucionales.
    """
    for layer in model.layers:
        # Verificar si la capa es convolucional
        if 'conv' in layer.name:
            weights, bias = layer.get_weights()
            print(f"{layer.name} tiene forma de filtro: {weights.shape}")

            # Normalizar los filtros para visualización
            f_min, f_max = weights.min(), weights.max()
            filters = (weights - f_min) / (f_max - f_min)

            # Seleccionar un filtro para visualizar
            filter_index = 0
            selected_filter = filters[:, :, :, filter_index]

            # Visualizar el filtro seleccionado
            fig, ax = plt.subplots()
            ax.imshow(selected_filter[:, :, 0], cmap='viridis')
            ax.set_title(f"{layer.name} - Filtro {filter_index}")
            plt.show()

def preprocess_image(image_path):
    """
    Preprocesa una imagen para ser compatible con el modelo.

    Argumentos:
    - image_path: Una cadena, la ruta al archivo de imagen.

    Retorna:
    - img_tensor_preprocesado: Un tensor de imagen preprocesado listo para ser utilizado por el modelo.
    """
    # Cargar y redimensionar la imagen
    img = load_img(image_path, target_size=(256, 256))

    # Convertir la imagen a un array de numpy
    img_array = img_to_array(img)

    # Añadir una dimensión de batch
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocesar la imagen
    img_tensor_preprocesado = preprocess_input(img_batch)

    return img_tensor_preprocesado

def visualize_activations(model, img_tensor):
    """
    Visualiza las activaciones de las capas convolucionales para una imagen dada.

    Argumentos:
    - model: Una instancia de tf.keras.Model.
    - img_tensor: Un tensor de imagen preprocesado.

    Retorna:
    Ninguno. Muestra las activaciones de las capas convolucionales para la imagen dada.
    """
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    for layer_name, layer_activation in zip([layer.name for layer in model.layers if 'conv' in layer.name], activations):
        filter_index = 0
        specific_activation = layer_activation[0, :, :, filter_index]

        plt.matshow(specific_activation, cmap='viridis')
        plt.title(f"Activación de {layer_name} - Filtro {filter_index}")
        plt.show()
