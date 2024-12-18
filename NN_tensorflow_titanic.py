import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --- Step 1: Data Preparation ---
# Load Titanic dataset
def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)

    # Select relevant features and target
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"
    data = data[features + [target]]

    # Handle missing values
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Convert categorical variables to numerical
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    # Split into features and labels
    X = data[features].values
    y = data[target].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


# Split dataset into training and testing sets
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 2: Neural Network Implementation with TensorFlow ---
# Build the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1)

# --- Step 3: Evaluation and Visualization ---
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot training accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
