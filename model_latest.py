
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv("train.csv")

X, Y = data.drop(['fraude', 'index'], axis=1), data["fraude"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100, stratify=Y)

X_train_noAnomaly = X_train[Y_train == 0]
Y_train_noAnomaly = Y_train[Y_train == 0]
X_test_withAnomalies = X_test
Y_test_withAnomalies = Y_test

class AnomalyDetector(Model):
    def __init__(self, l1_value=0.000001):
        super(AnomalyDetector, self).__init__()
        self.reductionDim = 8
        self.encoder = tf.keras.Sequential([
            layers.Dense(29, activation="relu" ),  
            layers.Dense(16, activation="relu"),  
            layers.Dense(self.reductionDim, activation="relu")
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(self.reductionDim,activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(29, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AnomalyDetector()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


autoencoder.fit(X_train_noAnomaly, X_train_noAnomaly,
                epochs=30,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[early_stopping])


X_test_WA_pred = autoencoder.predict(X_test_withAnomalies)
mse = np.mean(np.power(X_test_withAnomalies - X_test_WA_pred, 2), axis=1)


print(f'Average reconstruction error (train): {np.mean(mse)}')


plt.subplot(1, 2, 1)
plt.hist(mse, bins=50, alpha=0.6,range=(0,0.001), color='blue')
plt.title('Reconstruction error distribution (Train)')
plt.xlabel('Reconstruction error')
plt.ylabel('Frequency')


thresholds = np.linspace(0, 0.1, 100)
f1_scores = []
for thresh in thresholds:
    preds = [1 if e > thresh else 0 for e in mse]
    f1 = f1_score(Y_test, preds)
    f1_scores.append(f1)


max_f1 = max(f1_scores)
max_f1_threshold = thresholds[f1_scores.index(max_f1)]


max_f1_threshold


predictions = [1 if e > max_f1_threshold else 0 for e in mse]


(len(predictions)),len(Y_test_withAnomalies)


accuracy = accuracy_score(Y_test_withAnomalies, predictions)
f1 = f1_score(Y_test_withAnomalies, predictions)
precision = precision_score(Y_test_withAnomalies, predictions)
recall = recall_score(Y_test_withAnomalies, predictions)

print(f'Maximum F1 Score: {max_f1}')
print(f'Best Threshold: {max_f1_threshold}')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')


predictionsDf = pd.DataFrame(X_test_WA_pred)


predictionsDf["Mse"] = mse
predictionsDf["Prediction"] = predictions
predictionsDf["True"] = list(Y_test_withAnomalies)


predictionsDf.to_csv("predictions.csv")


