import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time

def show_result(X_names, y, y_pred):

    sub = pd.DataFrame()
    sub['Names'] = X_names
    sub['Predict_Type'] = y_pred + 1
    sub['Origin_Type'] = y + 1
    sub['Correct'] = [True if y_pred[i] == y[i] else False for i in range(len(y))]

    fig, ax = plt.subplots(figsize=(20,10))
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=sub.values,colWidths = [0.25]*len(sub.columns),
          rowLabels=sub.index,
          colLabels=sub.columns,
          cellLoc = 'center', rowLoc = 'center',
          loc='center')

    fig.tight_layout()

    plt.show()


def plot_confusion_matrix(cm):

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(range(7), range(1, 8))
    plt.yticks(range(7), range(1, 8))


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True value')
    plt.xlabel('Predicted value')
    plt.tight_layout()

    plt.show()
    plt.close()


def preprocess(dataset):
    # Izdvajamo poslednji red, klasu
    y = dataset.iloc[:, -1].values

    # S obzirom da imamo 7 razlicitih klasa, zelimo da imamo 7 izlaza, za svaku instancu jedan izlaz ce imati vrednost 1, a ostali 0, takozvani one-hot bitovi
    onehotencoder = OneHotEncoder()
    y = onehotencoder.fit_transform(y.reshape(-1, 1)).toarray()
    y = np.asarray(y, dtype=int)

    X = dataset.iloc[:, :17].values
    # Mozemo ovo primeniti i na broj nogu, u tom slucaju bismo imali vise ulaza u mrezu
    # legs = onehotencoder.fit_transform(X[:, 13].reshape(-1, 1)).toarray()
    # legs = np.asarray(legs, dtype=int)
    # Xnew = np.append(X, legs, axis=1)
    # X = np.delete(Xnew, 13, axis=1)

    return X, y

def main():
    start_time = time.time()

    # Ucitavanje baze
    ds = pd.read_csv("./zoo-animal-classification/zoo.csv")

    # Pretprocesiranje podaraka,
    # delimo na ulazne i izlazne podatke
    X, y = preprocess(ds)

    # Delimo podatke na dva skupa,
    # za ucenje i za testiranje
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    # Imena je jedinstveno za svaku instancu, zbog taga ga necemo koristiti u ucenju mreze
    # pamtimo ih radi lepse vizualizacije
    X_train_names = X_train[:, 0]
    X_test_names = X_test[:, 0]
    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    # Inicijalizacija neuronske mreze
    model = Sequential()

    # Dodajemo slojeve
    # Aktivaciona funkcija f(x) = max(x, 0)
    model.add(Dense(units=7, init='uniform', activation='relu', input_dim=16))
    model.add(Dense(units=7, init='uniform', activation='sigmoid'))

    # Kompajliramo mrezu
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

    # Pokrecemo ucenje mreze
    model.fit(X_train, y_train, batch_size=16, epochs=2000)

    # Na izlazu dobijamo niz od 7 elemenata, jedna jedinica i 6 nula, indeks jedinice predstavlja grupu kojoj instanca pripada -1
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Uzimamo indekse za svaku instancu, oni predstavljaju klasu kojoj instanca pripada
    # Klase instanci koje im dodeljuje mreza
    y_pred_train = np.argmax(y_pred_train, axis=1)
    y_pred_test = np.argmax(y_pred_test, axis=1)

    # Originalne klase instanci
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Pravimo matricu konfuzije kako bismo videli koliko je instanci dobro klasifikovano
    cm = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    print(sum(cm_test[i, i] for i in range(len(cm_test[0])))/(sum(sum(cm_test))+0.0))

    print(cm)
    print(cm_test)

    plot_confusion_matrix(cm_test)

    show_result(X_test_names, y_test, y_pred_test)

    print(time.time() - start_time)

if __name__ == '__main__':
    main()
