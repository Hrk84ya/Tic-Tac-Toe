document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const board = document.getElementById('board');
    const statusDisplay = document.getElementById('status');
    const probabilityDisplay = document.getElementById('probability');
    const resetButton = document.getElementById('reset-btn');
    const difficultySelect = document.getElementById('difficulty');
    
    // Game state
    const cells = [];
    let gameActive = true;
    let currentPlayer = 'x';
    let gameState = ['', '', '', '', '', '', '', '', ''];
    const winningConditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
        [0, 4, 8], [2, 4, 6] // diagonals
    ];

    // Create the game board
    function createBoard() {
        board.innerHTML = '';
        cells.length = 0;
        gameState = ['', '', '', '', '', '', '', '', ''];
        
        for (let i = 0; i < 9; i++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.setAttribute('data-cell-index', i);
            cell.addEventListener('click', handleCellClick);
            board.appendChild(cell);
            cells.push(cell);
        }
    }

    // Handle cell click
    function handleCellClick(e) {
        const clickedCell = e.target;
        const clickedCellIndex = parseInt(clickedCell.getAttribute('data-cell-index'));

        if (gameState[clickedCellIndex] !== '' || !gameActive) {
            return;
        }

        handlePlayerMove(clickedCell, clickedCellIndex);
    }

    // Handle player move
    function handlePlayerMove(clickedCell, clickedCellIndex) {
        gameState[clickedCellIndex] = currentPlayer;
        clickedCell.textContent = currentPlayer.toUpperCase();
        clickedCell.classList.add(currentPlayer);
        clickedCell.classList.add('disabled');

        updateWinProbability();
        
        if (checkWin()) {
            handleWin();
            return;
        }

        if (checkDraw()) {
            handleDraw();
            return;
        }

        // Switch to computer's turn
        currentPlayer = 'o';
        statusDisplay.textContent = "Computer's turn (O)";
        
        // Computer makes a move after a short delay
        setTimeout(computerMove, 500);
    }

    // Get current difficulty level
    function getDifficulty() {
        return difficultySelect.value;
    }

    // Computer makes a move based on difficulty
    function computerMove() {
        const emptyCells = gameState.map((cell, index) => cell === '' ? index : null).filter(val => val !== null);
        if (emptyCells.length === 0) return;

        let cellIndex;
        const difficulty = getDifficulty();
        
        switch(difficulty) {
            case 'easy':
                // Easy: Random moves
                cellIndex = emptyCells[Math.floor(Math.random() * emptyCells.length)];
                break;
                
            case 'medium':
                // Medium: 50% chance to make a smart move, 50% random
                if (Math.random() > 0.5) {
                    cellIndex = findBestMove();
                } else {
                    cellIndex = emptyCells[Math.floor(Math.random() * emptyCells.length)];
                }
                break;
                
            case 'hard':
                // Hard: Always make the best move
                cellIndex = findBestMove();
                break;
                
            default:
                cellIndex = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        }

        // If no best move found (shouldn't happen), pick random
        if (cellIndex === null || cellIndex === undefined) {
            cellIndex = emptyCells[Math.floor(Math.random() * emptyCells.length)];
        }

        // Make the move
        gameState[cellIndex] = 'o';
        const cell = cells[cellIndex];
        cell.textContent = 'O';
        cell.classList.add('o');
        cell.classList.add('disabled');

        updateWinProbability();
        
        if (checkWin()) {
            handleWin();
            return;
        }

        if (checkDraw()) {
            handleDraw();
            return;
        }

        currentPlayer = 'x';
        statusDisplay.textContent = "Your turn (X)";
    }
    
    // Find the best move using minimax algorithm
    function findBestMove() {
        let bestScore = -Infinity;
        let bestMove = null;
        
        // Check if there's an immediate win
        for (let i = 0; i < 9; i++) {
            if (gameState[i] === '') {
                gameState[i] = 'o';
                if (checkWin()) {
                    gameState[i] = ''; // Undo move
                    return i;
                }
                gameState[i] = ''; // Undo move
            }
        }
        
        // Check if we need to block opponent's win
        for (let i = 0; i < 9; i++) {
            if (gameState[i] === '') {
                gameState[i] = 'x';
                if (checkWin()) {
                    gameState[i] = ''; // Undo move
                    return i; // Block the win
                }
                gameState[i] = ''; // Undo move
            }
        }
        
        // Otherwise, use minimax to find best move
        for (let i = 0; i < 9; i++) {
            if (gameState[i] === '') {
                gameState[i] = 'o';
                let score = minimax(gameState, 0, false);
                gameState[i] = ''; // Undo move
                if (score > bestScore) {
                    bestScore = score;
                    bestMove = i;
                }
            }
        }
        
        return bestMove !== null ? bestMove : emptyCells[Math.floor(Math.random() * emptyCells.length)];
    }
    
    // Minimax algorithm for optimal moves
    function minimax(board, depth, isMaximizing) {
        // Check terminal states
        if (checkWin()) {
            return isMaximizing ? -10 + depth : 10 - depth;
        } else if (checkDraw()) {
            return 0;
        }
        
        if (isMaximizing) {
            let bestScore = -Infinity;
            for (let i = 0; i < 9; i++) {
                if (board[i] === '') {
                    board[i] = 'o';
                    const score = minimax(board, depth + 1, false);
                    board[i] = ''; // Undo move
                    bestScore = Math.max(score, bestScore);
                }
            }
            return bestScore;
        } else {
            let bestScore = Infinity;
            for (let i = 0; i < 9; i++) {
                if (board[i] === '') {
                    board[i] = 'x';
                    const score = minimax(board, depth + 1, true);
                    board[i] = ''; // Undo move
                    bestScore = Math.min(score, bestScore);
                }
            }
            return bestScore;
        }
    }

    // Check for a win
    function checkWin() {
        return winningConditions.some(condition => {
            const [a, b, c] = condition;
            return gameState[a] !== '' && 
                   gameState[a] === gameState[b] && 
                   gameState[a] === gameState[c];
        });
    }

    // Check for a draw
    function checkDraw() {
        return !gameState.includes('');
    }

    // Handle win
    function handleWin() {
        const winner = currentPlayer === 'x' ? 'You' : 'Computer';
        statusDisplay.textContent = `${winner} win${currentPlayer === 'x' ? '!' : 's!'}`;
        highlightWinningCells();
        gameActive = false;
    }

    // Handle draw
    function handleDraw() {
        statusDisplay.textContent = "Game ended in a draw!";
        gameActive = false;
    }

    // Highlight winning cells
    function highlightWinningCells() {
        const winningCondition = winningConditions.find(condition => {
            const [a, b, c] = condition;
            return gameState[a] !== '' && 
                   gameState[a] === gameState[b] && 
                   gameState[a] === gameState[c];
        });

        if (winningCondition) {
            winningCondition.forEach(index => {
                cells[index].classList.add('winner');
            });
        }
    }

    // Update win probability by calling the backend API
    async function updateWinProbability() {
        // Show loading state
        probabilityDisplay.textContent = 'Calculating...';
        
        try {
            // Use relative URL for better compatibility
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    board_state: gameState
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                const probability = data.probability;
                const percentage = (probability * 100).toFixed(1);
                probabilityDisplay.textContent = `Win probability for X: ${percentage}%`;
                
                // Visual indicator based on probability
                probabilityDisplay.style.color = getProbabilityColor(probability);
            } else {
                throw new Error(data.error || 'Failed to get prediction');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            console.error('Error details:', {
                message: error.message,
                stack: error.stack
            });
            // Fallback to simple heuristic if API fails
            const xCount = gameState.filter(cell => cell === 'x').length;
            const oCount = gameState.filter(cell => cell === 'o').length;
            let probability = 0.5;
            
            if (xCount > oCount) probability = 0.6;
            else if (xCount < oCount) probability = 0.4;
            
            probabilityDisplay.textContent = `Win probability for X: ${(probability * 100).toFixed(1)}% (estimate)`;
            probabilityDisplay.style.color = getProbabilityColor(probability);
        }
    }
    
    // Helper function to get color based on probability
    function getProbabilityColor(probability) {
        if (probability > 0.7) return '#2ecc71';  // Green for high probability
        if (probability > 0.4) return '#f1c40f';  // Yellow for medium
        return '#e74c3c';  // Red for low probability
    }

    // Reset game
    function resetGame() {
        gameActive = true;
        currentPlayer = 'x';
        const difficulty = getDifficulty();
        const difficultyText = difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
        statusDisplay.textContent = `Your turn (X) - ${difficultyText} difficulty`;
        createBoard();
        updateWinProbability();
    }

    // Event listeners
    resetButton.addEventListener('click', resetGame);
    difficultySelect.addEventListener('change', resetGame);

    // Initialize the game
    createBoard();
    updateWinProbability();
});