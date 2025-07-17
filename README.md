# Tic-Tac-Toe Win Predictor

This project implements a machine learning model to predict whether a given tic-tac-toe board configuration results in a win for 'x' (who moves first).

## Dataset
The dataset contains all possible endgame board configurations for tic-tac-toe, where 'x' is assumed to have played first. Each board configuration is labeled as either a win for 'x' (positive) or not (negative).

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train and evaluate the model:
   ```bash
   python train_model.py
   ```

   This will:
   - Load the tic-tac-toe dataset
   - Split it into training and testing sets
   - Train a Random Forest classifier
   - Print the model's accuracy and classification report
   - Show an example prediction

## Model

The model uses a Random Forest classifier with the following features:
- Each of the 9 board positions (one-hot encoded as 'x', 'o', or 'b' for blank)
- The target is binary: 1 if 'x' wins, 0 otherwise

## Example

You can use the `predict_win()` function to check if a given board state is a win for 'x'. The function takes a list of 9 elements representing the board in row-major order and returns the probability of a win for 'x'.

```python
# Example: x wins in the first row
board = [
    'x', 'x', 'x',  # First row
    'o', 'o', 'b',   # Second row
    'b', 'b', 'b'    # Third row
]
prob = predict_win(board)
print(f"Probability of win for 'x': {prob:.2f}")
```
