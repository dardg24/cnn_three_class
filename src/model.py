from funciones import *
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import pickle
import seaborn as sns

# Declaro primero las constantes que usaré durante el desarrollo
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANELS = 3
EPOCHS = 50
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
N_CLASS = 3

# conjunto de entrenamiento
train_dataset = tf.keras.utils.image_dataset_from_directory(
    '../train',  
    shuffle=True,
    image_size=((IMAGE_SIZE,IMAGE_SIZE)),
    batch_size=BATCH_SIZE
)

# conjunto de validación
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    '../validation',  
    shuffle=True,
    image_size=((IMAGE_SIZE,IMAGE_SIZE)),
    batch_size=BATCH_SIZE
)

# conjunto de prueba
test_dataset = tf.keras.utils.image_dataset_from_directory(
    '../test',  
    shuffle=True,
    image_size=((IMAGE_SIZE,IMAGE_SIZE)),
    batch_size=BATCH_SIZE
)

class_name = train_dataset.class_names
print (class_name)


print (len(train_dataset))
print (len(validation_dataset))
print (len(test_dataset))

# 32 es la cantidad de imagenes por batch
print (281 *  32)
print (36 *  32)
print (36 *  32)

#Visualizo un poco de las imagenes que estan en la primer batch
for image_batch, label_batch in train_dataset.take(1):
    for i in range(10):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.axis('off')
        plt.title(class_name[label_batch[i]])
    plt.show()
    print(f"Las dimensiones de las batch de imagenes son: {image_batch.shape}, y las clases serían: {label_batch.numpy()}")
    

# Preparo los datos mejorando el rendimiento durante el entrenamiento de forma eficiente, reduzco el tiempo de espera y aprovecho los recursos disponibles
train_dataset = train_dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)

# Reescalo las imagenes así mantengo un formato para todas y además las normalizo
rescale_resize_layers = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])

data_augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(factor=(0.2, 0.5)),
    # Este layer lo añadí después de experimentar con distintas arquitecturas del modelo buscando un mejor rendimiento
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='reflect')
])


# Voy a dejar comentada la arquitectura del modelo final, en el informe explicaré el abordaje utilizado
# También voy a enviar los notebooks utilizados desde el data collection hasta las pruebas finales
"""
Modelo Final:

new_model = models.Sequential([
    rescale_resize_layers,
    data_augmentation_layer,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    

    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  
    

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(N_CLASS, activation='softmax')
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  
    patience=10,  
    restore_best_weights=True  
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='new_model.h5', 
    monitor='val_loss',  
    save_best_only=True,  
    mode='min',  
    save_weights_only=False,  
    verbose=1  

new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

def adjusted_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 14:
        return lr * np.exp(-0.1)
    else:
        # Mantén el learning rate constante después de la epoch 14
        return 6.2500e-05

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(adjusted_scheduler)
callbacks=[early_stop, model_checkpoint,lr_scheduler]

new_history = new_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,  
    callbacks=callbacks
)

"""

# Para comprobar los resultados cargaré el modelo entrenado

model_path = '../models/new_model.h5'
model = load_model(model_path)


print (model.summary())

# Me traigo el history del entrenamiento para luego hacer plot del entrenamiento según las epoch
with open('../models/model_history.pkl', 'rb') as file:
    history = pickle.load(file)

plot_training_history(history)

# Procedo a evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_dataset)
print (f'Test Loss:{test_loss},\n Test Accuracy: {test_accuracy}')

# Una predicción de todo el conjunto de prueba
y_pred = model.predict(test_dataset)

# Y las clases
y_pred_classes = np.argmax(y_pred, axis=1)

# Obtenemos las etiquetas reales del conjunto de prueba
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Calculamos las métricas de clasificación
print(classification_report(y_true, y_pred_classes))

# Calculamos y mostramos la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Graficamos la matriz de confusión
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_name, yticklabels=class_name)
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Verdaderas')
plt.title('Matriz de Confusión')
plt.show()

for images, labels in test_dataset.take(1):
    # Hacemos predicciones
    preds = model.predict(images)
    preds_classes = np.argmax(preds, axis=1)
    
    # Buscamos índices de predicciones incorrectas
    incorrect_indices = np.where(preds_classes != labels.numpy())[0]
    correct_indices = np.where(preds_classes == labels.numpy())[0]
    
    # Ajustamos para mostrar hasta 6 imágenes incorrectas y 6 correctas
    N = 6
    
    plt.figure(figsize=(20, 8))
    
    # Mostramos imágenes incorrectas
    for i, incorrect_index in enumerate(incorrect_indices[:N]):
        plt.subplot(2, N, i + 1)  # Ajustamos para dos filas y N columnas
        plt.imshow(images[incorrect_index].numpy().astype("uint8"))
        plt.title(f'Pred: {preds_classes[incorrect_index]}, Real: {labels[incorrect_index].numpy()}')
        plt.axis('off')
    
    # Mostramos imágenes correctas
    for i, correct_index in enumerate(correct_indices[:N]):
        plt.subplot(2, N, N + i + 1)  # Continuamos en la segunda fila
        plt.imshow(images[correct_index].numpy().astype("uint8"))
        plt.title(f'Pred: {preds_classes[correct_index]} = Real: {labels[correct_index].numpy()}')
        plt.axis('off')
    
    plt.show()

visualize_filters(model)

image_path = '../test/birds/bird_3212.jpg'
image = preprocess_image(image_path)
visualize_activations(model, image)