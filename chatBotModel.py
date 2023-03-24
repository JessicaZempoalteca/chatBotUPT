#procesar las palabras que escribe el usuario
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

palabras = []
clases = []
docs = []
# arreglo que contiene las palabras que no se van a tomar en cuenta para el entrenamiento
gramaticaIgnorada = ['¿', '?', '!', '¡', ',', '.']
# cargar el archivo json con las preguntas y respuestas
baseConocimiento = open('steps.json', encoding='utf-8').read()
data = json.loads(baseConocimiento)

for intent in data['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        palabras.extend(w)

        docs.append((w, intent['tag']))

        if intent['tag'] not in clases:
            clases.append(intent['tag'])
            
            
words = [lemmatizer.lemmatize(w.lower()) for w in palabras if w not in gramaticaIgnorada]
words = sorted(list(set(words)))

classes = sorted(list(set(clases)))

print (len(docs), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in docs:

    bag = []

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)


    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    

random.shuffle(training)
training = np.array(training)
# creacion de los datos de entrenamiento y de prueba
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Datos de entrenamiento creados")

# CREACION DEL MODELO
# Contiene 3 capas:
# 1. Capa de entrada con 128 neuronas
# 2. Capa oculta con 64 neuronas
# 3. Capa de salida con tantas neuronas como clases (intents) y una función de activación softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
# Funcion de activacion relu para las capas ocultas
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Funcion de activacion softmax para la capa de salida
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo. Stochastic gradient descent con Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamiento del modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5', hist)

print("MODELO CREADO CORRECTAMENTE")