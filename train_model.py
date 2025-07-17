import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from arff2pandas import a2p

# Load the dataset
with open('Dataset.arff') as f:
    df = a2p.load(f)

# The last column is the target
X = df.iloc[:, :-1]  # All columns except the last one
# Map the target to binary (1 for positive, 0 for negative)
y = (df.iloc[:, -1] == 'positive').astype(int)

# Convert categorical data to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, columns=X.columns)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Function to predict win for a given board state
def predict_win(board_state):
    """
    Predict if the given board state is a win for 'x'.
    
    Args:
        board_state: List of 9 elements representing the board state in row-major order.
                    'x' for x, 'o' for o, 'b' for blank.
                    
    Returns:
        Probability of win for 'x' (between 0 and 1).
    """
    # Convert the input to a DataFrame with the same structure as training data
    input_df = pd.DataFrame([board_state], columns=X.columns)
    # One-hot encode the input
    input_encoded = pd.get_dummies(input_df)
    
    # Make sure the input has the same columns as the training data
    # Add missing columns with 0s
    for col in X_encoded.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[X_encoded.columns]
    
    # Predict probability
    prob = model.predict_proba(input_encoded)[0][1]  # Probability of positive class (win for x)
    return prob

# Example usage
if __name__ == "__main__":
    # Example board state (x wins in first row)
    example_board = ['x', 'x', 'x',  # First row
                    'o', 'o', 'b',   # Second row
                    'b', 'b', 'b']   # Third row
    
    prob = predict_win(example_board)
    print(f"\nExample prediction for board {example_board}:")
    print(f"Probability of win for 'x': {prob:.4f}")
