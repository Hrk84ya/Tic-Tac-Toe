# Tic-Tac-Toe AI with Win Prediction

A web-based Tic-Tac-Toe game featuring an AI opponent with multiple difficulty levels and real-time win probability prediction.

## Features

- 🎮 Play against an AI with three difficulty levels:
  - **Easy**: Makes random moves
  - **Medium**: Mix of random and strategic moves
  - **Hard**: Uses Minimax algorithm for optimal play
- 📊 Real-time win probability display
- 🎨 Modern, responsive web interface
- 🤖 Machine learning model for win prediction

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tic
   ```

2. **Build and start the application**:
   ```bash
   # For development (with hot-reload):
   docker-compose up dev

   # For production:
   docker-compose up web
   ```

3. **Access the application**:
   ```
   http://localhost:9000
   ```

### Local Development (Without Docker)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tic
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the development server**:
   ```bash
   ./start.sh --dev
   ```

5. **Open in your browser**:
   ```
   http://localhost:9000
   ```

## How to Play

1. Select your preferred difficulty level from the dropdown
2. Click "New Game" to start
3. You play as X (first player)
4. The AI will automatically make its move as O
5. Try to get three in a row to win!

## Project Structure

- `app.py` - Flask web server and API endpoints
- `train_model.py` - Machine learning model training and prediction logic
- `static/` - Frontend files
  - `index.html` - Game interface
  - `styles.css` - Styling
  - `script.js` - Game logic and AI implementation
- `Dataset.arff` - Training data
- `tictactoe_model.joblib` - Trained model (generated after first run)

## Technical Details

### AI Implementation

- **Easy**: Random moves
- **Medium**: 50% random moves, 50% strategic moves
- **Hard**: Minimax algorithm with alpha-beta pruning

### Win Prediction

The game features a machine learning model that predicts the probability of 'X' winning from any given board state. The model is a Random Forest classifier trained on all possible tic-tac-toe board configurations.

## Development

### Development Workflow

1. **Development Mode** (with hot-reload):
   ```bash
   docker-compose up dev
   ```
   - The server will automatically reload when you make changes to Python files
   - Static files are mounted as volumes for live updates

2. **Running Tests**:
   ```bash
   # Run tests inside the container
   docker-compose exec dev python -m pytest
   ```

3. **Accessing the Container**:
   ```bash
   # Get a shell in the running container
   docker-compose exec dev bash
   ```

### Production Deployment

1. **Build and Run**:
   ```bash
   docker-compose build
   docker-compose up -d web
   ```

2. **View Logs**:
   ```bash
   docker-compose logs -f web
   ```

3. **Scaling**:
   ```bash
   docker-compose up -d --scale web=4
   ```

### Environment Variables

Configure the application using these environment variables:

- `FLASK_ENV`: Set to 'development' or 'production'
- `FLASK_DEBUG`: Enable/disable debug mode (1/0)
- `PYTHONUNBUFFERED`: Set to 1 for better logging

## License

This project is open source and available under the [MIT License](LICENSE).
