import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'tictactoe_model.joblib'
COLUMNS = [
    'top-left', 'top-center', 'top-right',
    'middle-left', 'middle-center', 'middle-right',
    'bottom-left', 'bottom-center', 'bottom-right'
]

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the tic-tac-toe dataset.
    
    Returns:
        Tuple containing features (X) and target (y) data
    """
    data_file = Path('Dataset.arff')
    if not data_file.exists():
        logger.error(f"Dataset file not found: {data_file}")
        raise FileNotFoundError(f"Dataset file not found: {data_file}")
    
    try:
        logger.info("Loading dataset...")
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            # Skip the header and find the data section
            for line in f:
                if line.strip().lower().startswith('@data'):
                    break
            
            # Read the data rows
            for line in f:
                line = line.strip()
                if line and not line.startswith('@'):
                    # Remove any quotes and split by comma
                    row = [x.strip('\'" ') for x in line.split(',')]
                    data.append(row)
        
        # Create DataFrame from the parsed data
        df = pd.DataFrame(data, columns=COLUMNS + ['class'])
        
        # The last column is the target
        X = df.iloc[:, :-1]
        y = (df.iloc[:, -1] == 'positive').astype(int)
        
        logger.info(f"Loaded dataset with {len(X)} samples")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def train_model() -> Tuple[Optional[RandomForestClassifier], Optional[List[str]]]:
    """
    Train and save the model.
    
    Returns:
        Tuple of (trained_model, feature_columns) or (None, None) on failure
    """
    try:
        # Load the data
        X, y = load_data()
        
        # Convert categorical data to numerical using one-hot encoding
        # Create all possible column combinations first to ensure consistent ordering
        all_possible_columns = []
        for col in COLUMNS:  # Use the global COLUMNS constant for consistent ordering
            for val in ['x', 'o', 'b']:
                all_possible_columns.append(f"{col}_{val}")
        
        # Apply one-hot encoding
        X_encoded = pd.get_dummies(X, columns=COLUMNS)
        
        # Add missing columns with 0s
        for col in all_possible_columns:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Ensure consistent column order
        X_encoded = X_encoded[all_possible_columns]
        
        # Save the column structure for later use
        COLUMN_ORDER = all_possible_columns
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=1   # Show progress during training
        )
        
        # Set feature names for the model
        model.fit(X_train, y_train)
        
        # Store feature names in the model (for scikit-learn >= 1.0)
        if hasattr(model, 'feature_names_in_'):
            model.feature_names_in_ = np.array(X_train.columns)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        
        # Save the model with all necessary information
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': model,
            'feature_columns': all_possible_columns,  # One-hot encoded column names
            'input_columns': COLUMNS,  # Original column names
            'accuracy': accuracy,
            'model_type': 'RandomForestClassifier',
            'version': '1.0.0'
        }
        
        # Save using the highest protocol for better compatibility
        joblib.dump(model_data, MODEL_PATH, protocol=joblib.highest_protocol)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        return model, all_possible_columns
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        return None, None

def load_saved_model() -> Tuple[Optional[RandomForestClassifier], Optional[List[str]]]:
    """
    Load the saved model and column order.
    
    Returns:
        Tuple of (model, feature_columns) or (None, None) if not found
    """
    try:
        if not MODEL_PATH.exists():
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return None, None
            
        logger.info(f"Loading model from {MODEL_PATH}")
        saved_data = joblib.load(MODEL_PATH)
        
        if not isinstance(saved_data, dict):
            logger.warning("Old model format detected. Please retrain the model.")
            return None, None
            
        model = saved_data.get('model')
        if model is None:
            logger.error("No model found in the saved data")
            return None, None
            
        # Get feature columns, default to COLUMNS if not found
        feature_columns = saved_data.get('feature_columns', COLUMNS)
        
        # Set feature names for the model if available (scikit-learn >= 1.0)
        if hasattr(model, 'feature_names_in_') and feature_columns:
            model.feature_names_in_ = np.array(feature_columns)
            
        logger.info("Model loaded successfully")
        return model, feature_columns
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None

# Try to load the model, if not found, train a new one
model, COLUMN_ORDER = load_saved_model()
if model is None:
    print("No saved model found. Training a new model...")
    model, COLUMN_ORDER = train_model()

def predict_win(board_state: List[str]) -> float:
    """
    Predict the probability of 'x' winning from the given board state.
    
    Args:
        board_state: List of 9 elements representing the board state in row-major order.
                    'x' for x, 'o' for o, 'b' for blank.
                    
    Returns:
        Probability of win for 'x' (between 0 and 1). Returns 0.5 if prediction fails.
    """
    global model, COLUMN_ORDER
    
    if not model:
        logger.warning("No model loaded for prediction")
        return 0.5  # Neutral probability if model isn't loaded
    
    if len(board_state) != 9:
        logger.error(f"Invalid board state length: {len(board_state)}. Expected 9.")
        return 0.5
    
    try:
        # Validate input values
        valid_values = {'x', 'o', 'b'}
        if not all(cell.lower() in valid_values for cell in board_state):
            invalid = [cell for cell in board_state if cell.lower() not in valid_values]
            logger.error(f"Invalid cell values: {invalid}")
            return 0.5
            
        # Convert to lowercase for consistency
        board_state = [cell.lower() for cell in board_state]
        
        # Create a DataFrame with the board state
        df = pd.DataFrame([board_state], columns=COLUMNS)
        
        # One-hot encode the input
        encoded_data = {}
        for col in COLUMNS:
            for val in ['x', 'o', 'b']:
                col_name = f"{col}_{val}"
                encoded_data[col_name] = [1 if df[col].iloc[0] == val else 0]
        
        # Create DataFrame with the exact column order expected by the model
        df_encoded = pd.DataFrame(encoded_data)
        
        # Ensure we have the exact same columns as during training
        # Get feature names from the model if available (for scikit-learn >= 1.0)
        if hasattr(model, 'feature_names_in_'):
            feature_columns = list(model.feature_names_in_)
            # Add any missing columns with 0s
            for col in feature_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            # Reorder columns to match training
            df_encoded = df_encoded[feature_columns]
        
        # Get the probability of the positive class (win for 'x')
        proba = model.predict_proba(df_encoded)[0][1]
        logger.debug(f"Prediction made. Probability of 'x' winning: {proba:.2f}")
        return float(proba)
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        return 0.5  # Fallback to neutral probability

if __name__ == "__main__":
    # Configure logging for the script
    logging.basicConfig(level=logging.INFO)
    
    # Example board state (x wins in first row)
    example_boards = [
        ['x', 'x', 'x', 'o', 'o', 'b', 'b', 'b', 'b'],  # x wins first row
        ['o', 'o', 'o', 'x', 'x', 'b', 'b', 'b', 'b'],  # o wins first row
        ['x', 'o', 'x', 'o', 'x', 'o', 'o', 'x', 'o'],  # draw
        ['x', 'o', 'b', 'b', 'x', 'o', 'b', 'b', 'x']   # x wins diagonal
    ]
    
    try:
        # Load or train the model
        if model is None:
            logger.info("No trained model found. Training a new model...")
            model, COLUMN_ORDER = train_model()
        
        if model is not None:
            logger.info("\nRunning example predictions...")
            for i, board in enumerate(example_boards, 1):
                prob = predict_win(board)
                print(f"\nExample {i} - Probability of 'x' winning: {prob:.2f}")
                print(f"{board[0]}|{board[1]}|{board[2]}")
                print(f"{board[3]}|{board[4]}|{board[5]}")
                print(f"{board[6]}|{board[7]}|{board[8]}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

    # Start the GUI if this script is run directly
    try:
        from tictactoe_gui import main as run_gui
        run_gui()
    except ImportError as e:
        print("GUI dependencies not available. Running in console mode only.")
