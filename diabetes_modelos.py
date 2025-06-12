
# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Clase para modelo de predicción
class ModeloPrediccion:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, test_size=0.25, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

    def train_model(self, model, param_distributions=None, n_splits=5, n_iter=10, random_state=42, scoring=None):
        self.model = model
        if param_distributions:  # Si hay parámetros, usar búsqueda
            from sklearn.model_selection import RandomizedSearchCV
            optim_model = RandomizedSearchCV(estimator=self.model,
                                             param_distributions=param_distributions,
                                             n_iter=n_iter,
                                             cv=n_splits,
                                             verbose=2,
                                             random_state=random_state,
                                             n_jobs=-1,
                                             scoring=scoring)
            optim_model.fit(self.X_train, self.y_train)
            self.model = optim_model.best_estimator_
            print(f"Mejores hiperparámetros: {optim_model.best_params_}")
            print(f"Mejor puntuación: {optim_model.best_score_}")
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate_model(modelo, X_test, y_test, nombre_modelo="Modelo"):
        """
        Imprime la matriz de confusión, precisión global y precisión por clase para un modelo dado.

        :param modelo: Modelo ya entrenado que implementa .predict().
        :param X_test: Datos de prueba (features).
        :param y_test: Etiquetas verdaderas de prueba.
        :param nombre_modelo: Nombre del modelo para mostrar en pantalla.
        """
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        precision_global = accuracy_score(y_test, y_pred)
        precision_por_clase = precision_score(y_test, y_pred, average=None)

        print(f"\nMatriz de confusión {nombre_modelo}:")
        print(cm)
        print(f"Precisión global: {precision_global:.2f}")
        print(f"Precisión por clase: {precision_por_clase}")

        
    def resumen_metricas_modelo(modelo, X_test, y_test, nombre_modelo="Modelo"):
        """
        Genera un DataFrame con las métricas principales de evaluación del modelo.

        :param modelo: Modelo ya entrenado que implementa .predict().
        :param X_test: Datos de prueba (features).
        :param y_test: Etiquetas verdaderas de prueba.
        :param nombre_modelo: Nombre del modelo que se incluirá en la columna 'Modelo'.

        :return: DataFrame con las columnas: Modelo, Precisión global, Error global, PP, PN
        """
        from sklearn.metrics import accuracy_score, precision_score

        y_pred = modelo.predict(X_test)
        precision_global = accuracy_score(y_test, y_pred)
        error_global = 1 - precision_global
        precision_por_clase = precision_score(y_test, y_pred, average=None)

        return pd.DataFrame([{
            "Modelo": nombre_modelo,
            "Precisión global": round(precision_global, 4),
            "Error global": round(error_global, 4),
            "PP": round(precision_por_clase[1], 4),
            "PN": round(precision_por_clase[0], 4)
        }])








