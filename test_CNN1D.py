# !pip install tensorflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten , Input

opened = []

csv_path = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CSV/'

masterlist = '/global/scratch/users/arvalcarcel/CSMUB/RESULTS/ALL_STATIONS_FINAL_REVISED.csv'

stations_df = pd.read_csv(masterlist)
station_num = stations_df['grdc_no']

arrayFile = [os.path.join(csv_path, f"{station_no}.csv") for station_no in station_num]

for file in arrayFile:
  df = pd.read_csv(file, index_col= None, header = 0)
  opened.append(df)

total_df = pd.concat(opened, axis = 0, ignore_index = True)
# print(total_df)

q = total_df['Q']
swe = total_df['SWE']
scaled = total_df['SWE_scaled']

# Convert q and swe to NumPy arrays
X = scaled.to_numpy()  # This will be 1D: shape (num_samples,)
y = q.to_numpy()

# Reshape X for Conv1D: (num_samples, num_features, channels)
X = X.reshape(len(X), -1, 1)  # Shape becomes (num_samples, 1 feature, 1 channel)
y = y.reshape(len(y), -1, 1)
# Normalize the data
# X = X.astype('float32') / np.max(X)  # Scale to [0, 1]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
CNN = Sequential()
CNN.add(Input(shape=(18446,1)))  # Adjust to match reshaped input
CNN.add(Conv1D(filters=18446, kernel_size=1, activation='relu'))  # Convolutional layer
CNN.add(Flatten())  # Flatten for the Dense layers
CNN.add(Dense(128, activation='relu'))  # Dense layer
CNN.add(Dense(1))  # Output layer

# Compile the model
CNN.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
CNNhistory = CNN.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

model.summary()

# Plot training and validation loss
plt.plot(CNN.history['loss'], label='Training Loss')
plt.plot(CNN.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CNN/loss_curve.png",bbox_inches = "tight")
# plt.show()

val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

plt.plot(CNN.history['accuracy'],'b')
plt.plot(CNN.history['val_accuracy'],'r')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f"/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CNN/model_epochs.png",bbox_inches = "tight")
# plt.show()

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Compare predictions with actual values
plt.scatter(y_val, y_pred, alpha=0.5)
plt.title('Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig(f"/global/scratch/users/arvalcarcel/CSMUB/RESULTS/CNN/predictions_actual.png",bbox_inches = "tight")
# plt.show()