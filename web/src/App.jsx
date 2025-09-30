import React, { useState } from 'react';
import Board from './components/Board';
import PlayerMenu from './components/PlayerMenu';
import './App.css';

function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [playerTypes, setPlayerTypes] = useState({ red: 'human', yellow: 'human' });

  const handlePlayersSelected = (types) => {
    setPlayerTypes(types);
    setGameStarted(true);
  };

  const handleGameReset = () => {
    setGameStarted(false);
  };

  return (
    <div className="app">
      <h1>Connect Four</h1>
      {!gameStarted ? (
        <PlayerMenu onPlayersSelected={handlePlayersSelected} />
      ) : (
        <Board 
          playerTypes={playerTypes} 
          onGameReset={handleGameReset}
        />
      )}
    </div>
  );
}

export default App;
