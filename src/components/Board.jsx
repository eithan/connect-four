import React, { useState } from 'react';
import Cell from './Cell';
import { ConnectFourGame } from '../game/connectFour.js';
import './Board.css';

function Board() {
  const [game] = useState(new ConnectFourGame());
  const [board, setBoard] = useState(Array(6).fill().map(() => Array(7).fill(null)));
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [gameStarted, setGameStarted] = useState(false);
  const [winner, setWinner] = useState(null);
  const [lastMove, setLastMove] = useState(null);

  const handleClick = (row, col) => {
    if (!gameStarted || winner) return;

    const move = game.makeMove(col);
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
    }
  };

  const startGame = () => {
    setGameStarted(true);
  };

  const resetGame = () => {
    game.reset();
    setBoard(Array(6).fill().map(() => Array(7).fill(null)));
    setCurrentPlayer(1);
    setWinner(null);
    setGameStarted(false);
    setLastMove(null);
  };

  const renderStatus = () => {
    if (!gameStarted) {
      return "Click 'Start Game' to begin!";
    }
    if (winner) {
      return `${winner === 1 ? 'Red' : 'Yellow'} wins!`;
    }
    return `${currentPlayer === 1 ? "Red's" : "Yellow's"} turn`;
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
        {!gameStarted ? (
          <button onClick={startGame}>Start Game</button>
        ) : (
          <button onClick={resetGame}>Reset Game</button>
        )}
      </div>
    </div>
  );
}

export default Board; 