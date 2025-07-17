from flask import Flask, jsonify, request, render_template, send_from_directory
import os
import logging
import joblib
from pathlib import Path
from train_model import predict_win, train_model, load_saved_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
    static_folder='static',
    template_folder='static'  # Serve HTML from static folder
)

# Load or train the model when the app starts
model = None
try:
    logger.info("Loading saved model...")
    model, _ = load_saved_model()
    
    if model is None:
        logger.info("No saved model found. Training a new model...")
        model, _ = train_model()
        
    if model is None:
        logger.error("Failed to load or train a model. Predictions will not be available.")
        
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the probability of 'x' winning from the given board state.
    
    Expected JSON payload:
    {
        "board_state": ["x", "o", "b", "b", "x", "o", "b", "b", "x"]
    }
    """
    try:
        data = request.get_json()
        if not data or 'board_state' not in data:
            return jsonify({
                'error': 'Missing board_state in request',
                'status': 'error'
            }), 400
            
        board_state = data['board_state']
        
        if len(board_state) != 9:
            return jsonify({
                'error': 'Board state must contain exactly 9 cells',
                'status': 'error'
            }), 400
        # Convert empty strings to 'b' for blank
        model_input = [cell if cell in ['x', 'o'] else 'b' for cell in board_state]
        
        try:
            prediction = predict_win(model_input)
            return jsonify({
                'probability': float(prediction),
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            
            # Fallback heuristic based on number of pieces
            x_count = model_input.count('x')
            o_count = model_input.count('o')
            
            if x_count > o_count:
                prob = min(0.9, 0.5 + (x_count - o_count) * 0.1)
            elif x_count < o_count:
                prob = max(0.1, 0.5 - (o_count - x_count) * 0.1)
            else:
                prob = 0.5
                
            return jsonify({
                'probability': prob,
                'status': 'fallback',
                'message': 'Using fallback heuristic',
                'x_count': x_count,
                'o_count': o_count
            })
        
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/<path:path>')
def static_file(path):
    """Serve static files from the static directory."""
    try:
        return send_from_directory('static', path)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {str(e)}")
        return jsonify({
            'error': 'File not found',
            'status': 'error'
        }), 404

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 9000))
    
    logger.info(f"Starting Flask server on port {port}...")
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    )