import React, { useState, useEffect } from 'react';
import Cell from './Cell';
import { ConnectFourGame } from '../game/connectFour.js';
import './Board.css';

// WinningLine component to draw a continuous line through winning cells
const WinningLine = ({ cells }) => {
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

  // Calculate positions as percentages of the board
  const cellWidth = 100 / 7; // 7 columns
  const cellHeight = 100 / 6; // 6 rows
  // Board has aspect-ratio 7/6 → height = width * (6/7)
  // Use this to convert height-percentage into width-percentage units for diagonal math
  const heightToWidthRatio = 6 / 7;

  let lineStyle = {};

  if (isHorizontal) {
    const startX = firstCell.column * cellWidth + cellWidth / 2;
    const endX = lastCell.column * cellWidth + cellWidth / 2;
    const y = firstCell.row * cellHeight + cellHeight / 2;

    lineStyle = {
      position: 'absolute',
      left: `${startX}%`,
      top: `${y}%`,
      width: `${endX - startX}%`,
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
    const x = firstCell.column * cellWidth + cellWidth / 2;
    const startY = firstCell.row * cellHeight + cellHeight / 2;
    const endY = lastCell.row * cellHeight + cellHeight / 2;

    lineStyle = {
      position: 'absolute',
      left: `${x}%`,
      top: `${startY}%`,
      width: '6px',
      height: `${endY - startY}%`,
      transform: 'translateX(-50%)',
      zIndex: 10,
    };

    return (
      <div className="winning-line" style={lineStyle}>
        <div className="winning-line-inner grow-y" />
      </div>
    );
  } else {
    // Diagonal: compute angle using width units for both axes to respect aspect ratio
    const startX = firstCell.column * cellWidth + cellWidth / 2;
    const endX = lastCell.column * cellWidth + cellWidth / 2;
    const startY = firstCell.row * cellHeight + cellHeight / 2;
    const endY = lastCell.row * cellHeight + cellHeight / 2;

    const deltaX = endX - startX; // percent of board width
    const deltaYHeightPct = endY - startY; // percent of board height
    const deltaYWidthPct = deltaYHeightPct * heightToWidthRatio; // convert to width-relative percent

    const lengthWidthPct = Math.sqrt(deltaX * deltaX + deltaYWidthPct * deltaYWidthPct);
    const angle = Math.atan2(deltaYWidthPct, deltaX) * (180 / Math.PI);

    lineStyle = {
      position: 'absolute',
      left: `${startX}%`,
      top: `${startY}%`,
      width: `${lengthWidthPct}%`,
      height: '6px',
      transform: `translateY(-50%) rotate(${angle}deg)` ,
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

function Board({ playerTypes = { red: 'human', yellow: 'human' }, onPlayersChanged = () => {}, ai = null }) {
  const [game] = useState(new ConnectFourGame());
  const [board, setBoard] = useState(Array(6).fill().map(() => Array(7).fill(null)));
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [winner, setWinner] = useState(null);
  const [lastMove, setLastMove] = useState(null);
  const [gameStarted, setGameStarted] = useState(false);
  const [winningCells, setWinningCells] = useState([]);

  const getRandomMove = async () => {
    if (!ai) {
      console.warn("AI not initialized, falling back to random move");
      const availableColumns = game.getAvailableColumns();
      if (availableColumns.length === 0) return null;
      return availableColumns[Math.floor(Math.random() * availableColumns.length)];
    }
    
    try {
      const aiMove = await ai.getMove(board);
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
      
      if (game.checkWin(move.row, move.column)) {
        setWinner(currentPlayer);
        setWinningCells(game.winningCells);
      } else {
        setCurrentPlayer(currentPlayer === 1 ? 2 : 1);
      }
      return true;
    }
    return false;
  };

  const handleClick = (row, col) => {
    if (winner) return;
    
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
      const timer = setTimeout(async () => {
        const randomColumn = await getRandomMove();
        if (randomColumn !== null) {
          makeMove(randomColumn);
        }
      }, 500); // Small delay to make it feel more natural
      
      return () => clearTimeout(timer);
    }
  }, [gameStarted, currentPlayer, winner, playerTypes]);

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
      return `${winner === 1 ? 'Red' : 'Yellow'} wins!`;
    }
    const currentPlayerType = currentPlayer === 1 ? playerTypes.red : playerTypes.yellow;
    const playerName = currentPlayer === 1 ? 'Red' : 'Yellow';
    const typeText = currentPlayerType === 'ai-alphazero' ? ' (AI-AlphaZero)' : '';
    return `${playerName}${typeText}'s turn`;
  };

  return (
    <div className="game-container">
      <div data-testid="game-status" className={`status ${winner ? `winner-${winner === 1 ? 'red' : 'yellow'}` : ''}`}>
        {renderStatus()}
      </div>
      <div className="board">
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
          <WinningLine cells={winningCells} />
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