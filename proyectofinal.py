# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:04:21 2023

@author: luis mercado
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Cargar el dataset
data = pd.read_csv('zoo.csv')

# Separar los datos en características (X) y la variable objetivo (y)
X = data.drop(['animal_name', 'class_type'], axis=1)  # Características
y = data['class_type']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Inicializar los clasificadores
logistic_reg = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear')
naive_bayes = GaussianNB()

# Entrenar los modelos
logistic_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)

# Realizar predicciones
y_pred_logistic = logistic_reg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_nb = naive_bayes.predict(X_test)

# Calcular las métricas para Logistic Regression
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, average='weighted')
recall_logistic = recall_score(y_test, y_pred_logistic, average='weighted')
f1_logistic = f1_score(y_test, y_pred_logistic, average='weighted')

# Calcular las métricas para K-Nearest Neighbors
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

# Calcular las métricas para Support Vector Machines
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Calcular las métricas para Naive Bayes
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

# Mostrar las métricas para cada modelo
print("Logistic Regression:")
print("Confusion Matrix:")
print(conf_matrix_logistic)
print(f"Accuracy: {accuracy_logistic}")
print(f"Precision: {precision_logistic}")
print(f"Sensitivity (Recall): {recall_logistic}")
print(f"F1 Score: {f1_logistic}")
print()

print("K-Nearest Neighbors:")
print("Confusion Matrix:")
print(conf_matrix_knn)
print(f"Accuracy: {accuracy_knn}")
print(f"Precision: {precision_knn}")
print(f"Sensitivity (Recall): {recall_knn}")
print(f"F1 Score: {f1_knn}")
print()

print("Support Vector Machines:")
print("Confusion Matrix:")
print(conf_matrix_svm)
print(f"Accuracy: {accuracy_svm}")
print(f"Precision: {precision_svm}")
print(f"Sensitivity (Recall): {recall_svm}")
print(f"F1 Score: {f1_svm}")
print()

print("Naive Bayes:")
print("Confusion Matrix:")
print(conf_matrix_nb)
print(f"Accuracy: {accuracy_nb}")
print(f"Precision: {precision_nb}")
print(f"Sensitivity (Recall): {recall_nb}")
print(f"F1 Score: {f1_nb}")
print()
