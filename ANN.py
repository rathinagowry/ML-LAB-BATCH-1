# Import libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Sample Training Data
# Hours studied, Hours slept
X = np.array([
    [2, 9],   # low study, high sleep
    [1, 5],   # very low study, medium sleep
    [3, 6],   # medium study, medium sleep
    [5, 1],   # high study, low sleep
    [10, 2],  # very high study, low sleep
    [7, 3],   # high study, medium sleep
], dtype=float)

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 1, 1, 1, 1])

# Normalize Input 
X = X / np.amax(X, axis=0)  # scale values between 0 and 1

# Build ANN Model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))   # Hidden layer with 4 neurons
model.add(Dense(1, activation='sigmoid'))             # Output layer (binary classification)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X, y, epochs=200, verbose=0)

# Test the Model
test_data = np.array([[4, 8], [8, 1]])  
test_data = test_data / np.amax(X, axis=0)  
predictions = model.predict(test_data)


for i, pred in enumerate(predictions):
    print(f"Student {i+1}: Pass Probability = {pred[0]:.4f} → Predicted = {'Pass' if pred[0] > 0.5 else 'Fail'}")