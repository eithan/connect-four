import React, { useState, useEffect, useRef } from 'react';
import Cell from './Cell';
import { ConnectFourGame } from '../game/connectFour.js';
import './Board.css';

// WinningLine component to draw a continuous line through winning cells
const WinningLine = ({ cells, boardRef }) => {
  if (cells.length < 2) return null;

  // Sort cells to determine line direction
  const sortedCells = [...cells].sort((a, b) => {
    if (a.row !== b.row) return a.row - b.row;
    return a.column - b.column;
  });

  const firstCell = sortedCells[0];
  const lastCell = sortedCells[sortedCells.length - 1];

  // Determine if it's horizontal, vertical, or diagonal
  const isHorizontal = firstCell.row === lastCell.row;
  const isVertical = firstCell.column === lastCell.column;
  const isDiagonal = !isHorizontal && !isVertical;

  // Compute exact pixel positions based on the board's real layout (padding + gaps)
  const boardEl = boardRef?.current;
  if (!boardEl) return null;

  const computed = getComputedStyle(boardEl);
  const paddingLeft = parseFloat(computed.paddingLeft) || 0;
  const paddingRight = parseFloat(computed.paddingRight) || 0;
  const paddingTop = parseFloat(computed.paddingTop) || 0;
  const paddingBottom = parseFloat(computed.paddingBottom) || 0;
  const columnGap = parseFloat(computed.columnGap || computed.gap) || 0;
  const rowGap = parseFloat(computed.rowGap || computed.gap) || 0;

  const columns = 7;
  const rows = 6;

  const contentWidth = boardEl.clientWidth - paddingLeft - paddingRight;
  const contentHeight = boardEl.clientHeight - paddingTop - paddingBottom;

  const cellWidthPx = (contentWidth - (columns - 1) * columnGap) / columns;
  const cellHeightPx = (contentHeight - (rows - 1) * rowGap) / rows;

  const centerOf = (row, col) => ({
    x: paddingLeft + col * (cellWidthPx + columnGap) + cellWidthPx / 2,
    y: paddingTop + row * (cellHeightPx + rowGap) + cellHeightPx / 2,
  });

  const startCenter = centerOf(firstCell.row, firstCell.column);
  const endCenter = centerOf(lastCell.row, lastCell.column);

  let lineStyle = {};

  if (isHorizontal) {
    const startX = startCenter.x;
    const endX = endCenter.x;
    const y = startCenter.y;

    lineStyle = {
      position: 'absolute',
      left: `${startX}px`,
      top: `${y}px`,
      width: `${endX - startX}px`,
      height: '6px',
      transform: 'translateY(-50%)',
      zIndex: 10,
    };

    return (
      <div className="winning-line" style={lineStyle}>
        <div className="winning-line-inner grow-x" />
      </div>
    );
  } else if (isVertical) {
    const x = startCenter.x;
    const startY = startCenter.y;
    const endY = endCenter.y;

    lineStyle = {
      position: 'absolute',
      left: `${x}px`,
      top: `${startY}px`,
      width: '6px',
      height: `${endY - startY}px`,
      transform: 'translateX(-50%)',
      zIndex: 10,
    };

    return (
      <div className="winning-line" style={lineStyle}>
        <div className="winning-line-inner grow-y" />
      </div>
    );
  } else {
    // Diagonal in pixel space
    const startX = startCenter.x;
    const startY = startCenter.y;
    const endX = endCenter.x;
    const endY = endCenter.y;

    const deltaX = endX - startX;
    const deltaY = endY - startY;

    const lengthPx = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const angle = Math.atan2(deltaY, deltaX) * (180 / Math.PI);

    lineStyle = {
      position: 'absolute',
      left: `${startX}px`,
      top: `${startY}px`,
      width: `${lengthPx}px`,
      height: '6px',
      transform: `translateY(-50%) rotate(${angle}deg)`,
      transformOrigin: '0 50%',
      zIndex: 10,
    };

    return (
      <div className="winning-line" style={lineStyle}>
        <div className="winning-line-inner grow-x" />
      </div>
    );
  }
};

function Board({ playerTypes = { red: 'human', yellow: 'human' }, onPlayersChanged = () => {}, ai = null, aiLoading = false, aiError = null, aiInitialized = false }) {
  const [game] = useState(new ConnectFourGame());
  const [board, setBoard] = useState(Array(6).fill().map(() => Array(7).fill(null)));
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [winner, setWinner] = useState(null);
  const [lastMove, setLastMove] = useState(null);
  const [gameStarted, setGameStarted] = useState(false);
  const [winningCells, setWinningCells] = useState([]);
  const boardRef = useRef(null);

  const getRandomMove = async () => {
    // Check if AI is loading or has error
    if (aiLoading) {
      console.warn("AI is still loading, waiting...");
      return null; // Return null to indicate we should wait
    }
    
    if (aiError) {
      console.warn("AI has error, falling back to random move");
      const availableColumns = game.getAvailableColumns();
      if (availableColumns.length === 0) return null;
      return availableColumns[Math.floor(Math.random() * availableColumns.length)];
    }
    
    if (!ai) {
      console.warn("AI not initialized, falling back to random move");
      const availableColumns = game.getAvailableColumns();
      if (availableColumns.length === 0) return null;
      return availableColumns[Math.floor(Math.random() * availableColumns.length)];
    }
    
    try {
      const aiMove = await ai.getMove(board, currentPlayer);
      console.log("*** AI move:", aiMove);
      return aiMove;
    } catch (error) {
      console.error("AI move failed, falling back to random move:", error);
      const availableColumns = game.getAvailableColumns();
      if (availableColumns.length === 0) return null;
      return availableColumns[Math.floor(Math.random() * availableColumns.length)];
    }
  };

  const makeMove = (column) => {
    const move = game.makeMove(column);
    if (move) {
      const newBoard = board.map(row => [...row]);
      newBoard[move.row][move.column] = currentPlayer;
      setBoard(newBoard);
      setLastMove({ row: move.row, column: move.column });

      // Trust the game engine to determine outcome (win or draw)
      const state = game.getState();
      if (state.winner) {
        if (state.winner === 'draw') {
          setWinner('draw');
        } else {
          setWinner(state.winner === 'red' ? 1 : 2);
        }
        setWinningCells(state.winningCells || []);
      } else {
        setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
      }
      return true;
    }
    return false;
  };

  const handleClick = (row, col) => {
    if (winner) return;
    
    // Block interactions while AI is loading
    if (aiLoading) return;
    
    const currentPlayerType = currentPlayer === 1 ? playerTypes.red : playerTypes.yellow;
    if (currentPlayerType !== 'human') return;

    // Start the game if it hasn't started yet
    if (!gameStarted) {
      setGameStarted(true);
    }

    makeMove(col);
  };

  // Handle AI-AlphaZero player moves
  useEffect(() => {
    if (!gameStarted || winner) return;
    
    const currentPlayerType = currentPlayer === 1 ? playerTypes.red : playerTypes.yellow;
    if (currentPlayerType === 'ai-alphazero') {
      // Only proceed if AI is initialized and not loading
      if (!aiInitialized || aiLoading) {
        return;
      }
      
      const timer = setTimeout(async () => {
        const randomColumn = await getRandomMove();
        if (randomColumn !== null) {
          makeMove(randomColumn);
        }
      }, 500); // Small delay to make it feel more natural
      
      return () => clearTimeout(timer);
    }
  }, [gameStarted, currentPlayer, winner, playerTypes, aiLoading, aiError, ai, aiInitialized]);

  // AI testing code moved to reusable function
  const testAIWithBoards = async () => {
    try {
      const ai = new ConnectFourAI("/connect-four/alphazero-network-model.onnx");
      await ai.init();

      // Test with different board states
      const emptyBoard = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]
      ];
      
      const testBoard1 = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0]
      ];
      
      const testBoard2 = [
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,-1,0,0,0,0],
        [0,0,-1,0,0,0,0],
        [0,1,-1,0,0,0,0],
        [0,1,1,0,0,0,1]
      ];
      
      console.log("Empty board:", emptyBoard);
      console.log("Test board 1:", testBoard1);
      console.log("Test board 2:", testBoard2);
      
      console.log("Testing empty board:");
      const move1 = await ai.getMove(emptyBoard);
      console.log("*********** Empty board move:", move1);
      
      console.log("Testing board with piece in column 0:");
      const move2 = await ai.getMove(testBoard1);
      console.log("*********** Board1 move:", move2);
      
      console.log("Testing board with piece in column 1:");
      const move3 = await ai.getMove(testBoard2);
      console.log("*********** Board2 move:", move3);
    } catch (error) {
      console.error("AI testing error:", error);
    }
  };

  const startGame = () => {
    setGameStarted(true);
  };

  const resetGame = async () => {
    game.reset();
    setBoard(Array(6).fill().map(() => Array(7).fill(null)));
    setCurrentPlayer(1);
    setWinner(null);
    setLastMove(null);
    setGameStarted(false);
    setWinningCells([]);
    
    // Reset AI synchronously if it exists
    if (ai) {
      try {
        // Reinitialize the AI to reset its state
        await ai.init();
        console.log('AI reset successfully');
      } catch (error) {
        console.error('Failed to reset AI:', error);
      }
    }
    
    // Automatically start the game after reset
    setGameStarted(true);
  };

  const handlePlayerTypeChange = (player, type) => {
    const newPlayerTypes = { ...playerTypes, [player]: type };
    onPlayersChanged(newPlayerTypes);
  };

  const renderStatus = () => {
    if (!gameStarted) {
      return "Select players and start!";
    }
    if (winner) {
      if (winner === 'draw') {
        return "It's a draw!";
      }
      return `${winner === 1 ? 'Red' : 'Yellow'} wins!`;
    }
    const currentPlayerType = currentPlayer === 1 ? playerTypes.red : playerTypes.yellow;
    const playerName = currentPlayer === 1 ? 'Red' : 'Yellow';
    
    if (currentPlayerType === 'ai-alphazero') {
      if (aiLoading) {
        return `${playerName} (AI-AlphaZero) is loading...`;
      }
      if (aiError) {
        return `${playerName} (AI-AlphaZero) - Error: Using random moves`;
      }
      if (!aiInitialized) {
        return `${playerName} (AI-AlphaZero) - Initializing...`;
      }
      return `${playerName} (AI-AlphaZero)'s turn`;
    }
    
    return `${playerName}'s turn`;
  };

  return (
    <div className="game-container">
      <div data-testid="game-status" className={`status ${winner ? (winner === 'draw' ? 'winner-draw' : `winner-${winner === 1 ? 'red' : 'yellow'}`) : ''}`}>
        {renderStatus()}
      </div>
      <div className="board" ref={boardRef}>
        {board.flatMap((row, rowIndex) => 
          row.map((cell, colIndex) => (
            <Cell
              key={`${rowIndex}-${colIndex}`}
              value={cell}
              onClick={() => handleClick(rowIndex, colIndex)}
              animate={!!lastMove && rowIndex === lastMove.row && colIndex === lastMove.column}
              isWinning={winningCells.some(cell => cell.row === rowIndex && cell.column === colIndex)}
            />
          ))
        )}
        {winningCells.length > 0 && (
          <WinningLine cells={winningCells} boardRef={boardRef} />
        )}
      </div>
      <div className="player-selection">
        <div className="player-select">
          <label>
            <span className="player-label red">Red Player</span>
            <select id="red-player-select"
              value={playerTypes.red} 
              onChange={(e) => handlePlayerTypeChange('red', e.target.value)}
            >
              <option value="human">Human</option>
              <option value="ai-alphazero">AI-AlphaZero</option>
            </select>
          </label>
        </div>
        <div className="player-select">
          <label>
            <span className="player-label yellow">Yellow Player</span>
            <select 
              value={playerTypes.yellow} 
              onChange={(e) => handlePlayerTypeChange('yellow', e.target.value)}
            >
              <option value="human">Human</option>
              <option value="ai-alphazero">AI-AlphaZero</option>
            </select>
          </label>
        </div>
      </div>
      <div className="controls">
        <button onClick={gameStarted ? () => resetGame() : startGame}>
          {gameStarted ? 'Reset Game' : 'Start Game'}
        </button>
      </div>
    </div>
  );
}

export default Board; 