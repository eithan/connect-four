import React, { useState, useEffect } from 'react';
import Cell from './Cell';
import { ConnectFourGame } from '../game/connectFour.js';
import './Board.css';

function Board({ playerTypes = { red: 'human', yellow: 'human' }, onGameReset = () => {} }) {
  const [game] = useState(new ConnectFourGame());
  const [board, setBoard] = useState(Array(6).fill().map(() => Array(7).fill(null)));
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [winner, setWinner] = useState(null);
  const [lastMove, setLastMove] = useState(null);

  const getRandomMove = () => {
    const availableColumns = game.getAvailableColumns();
    if (availableColumns.length === 0) return null;
    return availableColumns[Math.floor(Math.random() * availableColumns.length)];
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

    makeMove(col);
  };

  // Handle Minimax player moves
  useEffect(() => {
    if (winner) return;
    
    const currentPlayerType = currentPlayer === 1 ? playerTypes.red : playerTypes.yellow;
    if (currentPlayerType === 'minimax') {
      const timer = setTimeout(() => {
        const randomColumn = getRandomMove();
        if (randomColumn !== null) {
          makeMove(randomColumn);
        }
      }, 500); // Small delay to make it feel more natural
      
      return () => clearTimeout(timer);
    }
  }, [currentPlayer, winner, playerTypes]);

  const resetGame = () => {
    game.reset();
    setBoard(Array(6).fill().map(() => Array(7).fill(null)));
    setCurrentPlayer(1);
    setWinner(null);
    setLastMove(null);
    onGameReset();
  };

  const renderStatus = () => {
    if (winner) {
      return `${winner === 1 ? 'Red' : 'Yellow'} wins!`;
    }
    const currentPlayerType = currentPlayer === 1 ? playerTypes.red : playerTypes.yellow;
    const playerName = currentPlayer === 1 ? 'Red' : 'Yellow';
    const typeText = currentPlayerType === 'minimax' ? ' (Minimax)' : '';
    return `${playerName}${typeText}'s turn`;
  };

  return (
    <div className="game-container">
      <div data-testid="game-status" className="status">
        {renderStatus()}
      </div>
      <div className="board">
        {board.map((row, rowIndex) => (
          <div key={rowIndex} className="board-row">
            {row.map((cell, colIndex) => (
              <Cell
                key={colIndex}
                value={cell}
                onClick={() => handleClick(rowIndex, colIndex)}
                animate={!!lastMove && rowIndex === lastMove.row && colIndex === lastMove.column}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="controls">
        <button onClick={resetGame}>Reset Game</button>
      </div>
    </div>
  );
}

export default Board; 