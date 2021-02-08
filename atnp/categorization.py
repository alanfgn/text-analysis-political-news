from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn import svm
from sklearn import tree
from keras.models import Sequential
from keras import layers
import tensorflow as tf


test_size = 0.25
random_state = 42


def evaluating(y_test, y_pred):

    cfm = confusion_matrix(y_test, y_pred)
    clr = classification_report(y_test, y_pred)
    acs = accuracy_score(y_test, y_pred)

    return cfm, clr, acs


def generate_gaussian_nb(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)

    return gnb, (y_test, y_pred)


def generate_multinomial_nb(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    mnb = MultinomialNB()
    y_pred = mnb.fit(x_train, y_train).predict(x_test)

    return mnb, (y_test, y_pred)


def generate_complement_nb(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    cnb = ComplementNB()
    y_pred = cnb.fit(x_train, y_train).predict(x_test)

    return cnb, (y_test, y_pred)


def generate_bernoulli_nb(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    bnb = BernoulliNB()
    y_pred = bnb.fit(x_train, y_train).predict(x_test)

    return bnb, (y_test, y_pred)


def generate_categorical_nb(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    cnb = CategoricalNB()
    y_pred = cnb.fit(x_train, y_train).predict(x_test)

    return cnb, (y_test, y_pred)


def generate_svm(x, y, kernel="precomputed"):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    clf = svm.SVC(kernel=kernel)
    y_pred = clf.fit(x_train, y_train).predict(x_test)

    return clf, (y_test, y_pred)


def generate_tree(x, y):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    tr = tree.DecisionTreeClassifier()
    y_pred = tr.fit(x_train, y_train).predict(x_test)

    return tr, (y_test, y_pred)



def generate_nn(x, y, num_layers=100, activation='relu', loss='binary_crossentropy', optimizer='adam', **kwargs):

    le = LabelEncoder()
    le.fit(list(set(y)))

    y = le.transform(y)
    y = tf.keras.utils.to_categorical(y, len(le.classes_))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    input_dim = x.shape[1]
    model = Sequential()

    model.add(layers.Dense(num_layers, input_dim=input_dim, activation=activation))  
    
    if "num_hidden_layers" in kwargs:
        
        num_hidden = kwargs["num_hidden"] if "num_hidden" in kwargs else num_layers
        hidden_activation = kwargs["hidden_activation"] if "hidden_activation" in kwargs else activation

        for _ in range(kwargs["num_hidden_layers"]):
            model.add(layers.Dense(num_hidden, activation=hidden_activation))
    
    model.add(layers.Dense(len(le.classes_), activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    history = model.fit(x_train, y_train,
                        epochs=100,
                        verbose=False,
                        validation_data=(x_test, y_test),
                        batch_size=10)
    y_pred = model.predict(x_test)

    def destranformate(y):
        return le.inverse_transform(y.argmax(1))

    y_test = destranformate(y_test)
    y_pred = destranformate(y_pred)
    
    return model, history, (y_test, y_pred)
    