import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

data = pd.read_csv("train.csv")

X, Y = data.drop(['fraude', 'index', 'montant'], axis=1), data["fraude"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=100, stratify=Y)
smote = SMOTE(random_state=42)
X_train, Y_train = smote.fit_resample(X_train, Y_train)
X_train = X_train[Y_train == 0]
Y_train = Y_train[Y_train == 0]

input_dim = X_train.shape[1]
encoding_dim = 14  

input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

autoencoder = Model(input_layer, decoder_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)


thresholds = np.linspace(0, 1, 100)
f1_scores = []
for thresh in thresholds:
    preds = [1 if e > thresh else 0 for e in mse]
    f1 = f1_score(Y_test, preds)
    f1_scores.append(f1)

max_f1 = max(f1_scores)
max_f1_threshold = thresholds[f1_scores.index(max_f1)]

predictions = [1 if e > max_f1_threshold else 0 for e in mse]

accuracy = accuracy_score(Y_test, predictions)
f1 = f1_score(Y_test, predictions)
precision = precision_score(Y_test, predictions)
recall = recall_score(Y_test, predictions)

print(f'Maximum F1 Score: {max_f1}')
print(f'Best Threshold: {max_f1_threshold}')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
