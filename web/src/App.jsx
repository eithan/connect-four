import React, { useState, useEffect } from 'react';
import Board from './components/Board';
import ConnectFourAI from './components/AI.jsx';
import './App.css';

function App() {
  const [playerTypes, setPlayerTypes] = useState({ red: 'human', yellow: 'human' });
  const [ai, setAi] = useState(null);

  useEffect(() => {
    const initializeAI = async () => {
      try {
        const aiInstance = new ConnectFourAI("/connect-four/alphazero-network-model.onnx");
        await aiInstance.init();
        setAi(aiInstance);
        console.log('AI initialized successfully');
      } catch (error) {
        console.error('Failed to initialize AI:', error);
      }
    };

    initializeAI();
  }, []);

  const handlePlayersChanged = (types) => {
    setPlayerTypes(types);
  };

  return (
    <div className="app">
      <h1>Connect Four</h1>
      <Board 
        playerTypes={playerTypes} 
        onPlayersChanged={handlePlayersChanged}
        ai={ai}
      />
    </div>
  );
}

export default App;
