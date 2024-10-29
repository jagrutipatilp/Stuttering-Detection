import librosa
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Load the saved model
model = load_model('audio_clip_features_model.h5')

# Load the new audio file
audio, sr = librosa.load('s4.wav')

# Extract the MFCC features from the audio file
mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)

# Reshape the MFCC features to match the input shape of the model
mfccs = mfccs.reshape(1, -1)

# Predict the audio clip features
predictions = model.predict(mfccs)

# Print the predicted audio clip features
print("Predictions:", predictions)

# Define feature names and their corresponding predicted values
feature_names = [
    'Unsure', 'PoorAudioQuality', 'Prolongation', 'Block', 'SoundRep', 
    'WordRep', 'DifficultToUnderstand', 'Interjection', 'NoStutteredWords', 
    'NaturalPause', 'Music', 'NoSpeech'
]
feature_values = predictions[0]  # Assuming predictions are an array of values

# Define a threshold for small values
threshold = 1e-5

# Prepare data for pie chart
filtered_feature_names = [name for name, value in zip(feature_names, feature_values) if value > threshold]
filtered_feature_values = [value for value in feature_values if value > threshold]

# Add 'Other' category for very small values
other_value = np.sum([value for value in feature_values if value <= threshold])
filtered_feature_names.append('Other')
filtered_feature_values.append(other_value)

# Plotting the predictions as a bar graph
plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_values, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Predicted Audio Clip Features - Bar Graph')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plotting the predictions as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(filtered_feature_values, labels=filtered_feature_names, autopct='%1.1f%%', startangle=140)
plt.title('Predicted Audio Clip Features - Pie Chart')
plt.show()
