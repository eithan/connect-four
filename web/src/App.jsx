import React, { useState } from 'react';
import Board from './components/Board';
import './App.css';

function App() {
  const [playerTypes, setPlayerTypes] = useState({ red: 'human', yellow: 'human' });

  const handlePlayersChanged = (types) => {
    setPlayerTypes(types);
  };

  return (
    <div className="app">
      <h1>Connect Four</h1>
      <Board 
        playerTypes={playerTypes} 
        onPlayersChanged={handlePlayersChanged}
      />
    </div>
  );
}

export default App;
