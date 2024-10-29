import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset
df = pd.read_csv('final_dataset.csv')

# Remove the 'Name' and 'Show' columns
df = df.drop(['Name', 'Show'], axis=1)

# Split the data into input and output
X = df.iloc[:, 16:]  # MFCC features
y = df.iloc[:, 4:16]  # Audio clip features

# Normalize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('audio_clip_features_model.h5')
