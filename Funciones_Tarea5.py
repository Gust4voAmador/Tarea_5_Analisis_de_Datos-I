# Funciones usadas en el main
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Función para entrenar modelo
def train_model(model, X_train, y_train, param_distributions=None, n_splits=5, n_iter=10, random_state=42, scoring=None):
    """
    Entrena el modelo con los datos de entrenamiento.
    """
    optim_model = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=n_splits,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
        scoring=scoring
    )

    optim_model.fit(X_train, y_train)

    print(f"Mejores hiperparámetros: {optim_model.best_params_}")
    print(f"Mejor puntuación: {optim_model.best_score_}")

    return optim_model.best_estimator_

# Función para predecir
def predict(model, X_new, probabilities=False):
    """
    Realiza predicciones sobre nuevos datos.

    :param model: Modelo entrenado.
    :param X_new: Datos nuevos para predecir.
    :param probabilities: Si es True, devuelve probabilidades.
    :return: Predicciones del modelo.
    """
    if probabilities:
        return model.predict_proba(X_new)
    else:
        return model.predict(X_new)
    
# Función para evaluar
def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo utilizando el conjunto de prueba y devuelve las métricas.

    :param model: Modelo entrenado.
    :param X_test: Conjunto de prueba (features).
    :param y_test: Conjunto de prueba (targets reales).
    :return: Diccionario con métricas de desempeño.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def evaluar_metricas(nombre, y_test, y_pred):
    precision_global = precision_score(y_test, y_pred, average='weighted')
    error_global = 1 - precision_global
    pp = precision_score(y_test, y_pred, pos_label=1)
    pn = precision_score(y_test, y_pred, pos_label=0)

    return {
        "Modelo": nombre,
        "Precisión global": round(precision_global, 4),
        "Error global": round(error_global, 4),
        "PP": round(pp, 4),
        "PN": round(pn, 4)
    }

def correr_ejer_4():
    # Cargar los datos
    df = pd.read_csv("novatosNBA.csv", sep=",", header=0)

    # Eliminar columnas no numéricas o irrelevantes
    df = df.drop(columns=['Unnamed: 0', 'Player', 'Team', 'Conf'])

    # Dividir en variables predictoras y variable objetivo
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LDA
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)
    lda_pred = lda_model.predict(X_test)

    # QDA
    qda_model = QuadraticDiscriminantAnalysis()
    qda_model.fit(X_train, y_train)
    qda_pred = qda_model.predict(X_test)

    # Naive Bayes
    naive_model = GaussianNB()
    naive_model.fit(X_train, y_train)
    naive_pred = naive_model.predict(X_test)

    # Retornar predicciones y verdaderos
    return y_test, lda_pred, qda_pred, naive_pred