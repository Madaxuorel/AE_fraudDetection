import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
import optuna

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
    def __init__(self, reductionDim=8):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(29, activation="relu"),  
            layers.Dense(16, activation="relu"),  
            layers.Dense(reductionDim, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(reductionDim, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(29, activation="sigmoid")
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

def objective(trial):
    reductionDim = trial.suggest_int('reductionDim', 2, 32)
    autoencoder = AnomalyDetector(reductionDim=reductionDim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(X_train_noAnomaly, X_train_noAnomaly,
                    epochs=30, shuffle=True,
                    validation_data=(X_test, X_test),
                    callbacks=[early_stopping], verbose=0)  # Minimize console output
    
    X_test_WA_pred = autoencoder.predict(X_test_withAnomalies)
    mse = np.mean(np.power(X_test_withAnomalies - X_test_WA_pred, 2), axis=1)
    
    thresholds = np.linspace(0, 0.1, 100)
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if e > thresh else 0 for e in mse]
        f1 = f1_score(Y_test_withAnomalies, preds)
        f1_scores.append(f1)
    max_f1 = max(f1_scores)
    
    return max_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # Modify n_trials based on your computational resource availability

print("Best reductionDim:", study.best_params['reductionDim'])
print("Best F1 score:", study.best_value)
