# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler

# %%
# Load the data
data = pd.read_csv("train.csv")

# Separate the features and the target
X, Y = data.drop(['fraude', 'index'], axis=1), data["fraude"]

# %%
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100, stratify=Y)

# %%
X_train = X_train[Y_train == 0]
Y_train = Y_train[Y_train == 0]

# %%
pd.DataFrame(X_train)

# %%
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(29, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(29, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

# %%
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# %%
autoencoder.fit(X_train, X_train,
                epochs=5,
                shuffle=True,
                validation_data=(X_test, X_test))

# %%
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# %%
pd.DataFrame(X_test_pred)

# %%
print(f'Average reconstruction error (train): {np.mean(mse)}')

# %%
plt.subplot(1, 2, 1)
plt.hist(mse, bins=50, alpha=0.6,range=(0,0.001), color='blue')
plt.title('Reconstruction error distribution (Train)')
plt.xlabel('Reconstruction error')
plt.ylabel('Frequency')

# %%
thresholds = np.linspace(0, 0.1, 100)
f1_scores = []
for thresh in thresholds:
    preds = [1 if e > thresh else 0 for e in mse]
    f1 = f1_score(Y_test, preds)
    f1_scores.append(f1)

# %%
max_f1 = max(f1_scores)
max_f1_threshold = thresholds[f1_scores.index(max_f1)]

# %%
max_f1_threshold

# %%
mse

# %%
predictions = [1 if e > max_f1_threshold else 0 for e in mse]

# %%
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

# %% [markdown]
# tests

# %%
testdf = pd.DataFrame(columns=["real","preds"])

# %%
testdf["real"] = Y_test

# %%
testdf["preds"] = predictions

# %%
testdf.to_excel("testdf.xlsx")

# %%
predsDf = pd.DataFrame(predictions,columns=["preds"])

# %%
predsDf["base"] = Y_test



plt.show()