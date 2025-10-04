import React, { useState, useEffect } from 'react';
import Board from './components/Board';
import ConnectFourAI from './components/AI.jsx';
import './App.css';

function App() {
  const [playerTypes, setPlayerTypes] = useState({ red: 'human', yellow: 'human' });
  const [ai, setAi] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [aiRetryCount, setAiRetryCount] = useState(0);
  const [aiInitialized, setAiInitialized] = useState(false);

  const initializeAI = async (retryCount = 0) => {
    try {
      setAiLoading(true);
      setAiError(null);
      
      console.log(`Initializing AI (attempt ${retryCount + 1})...`);
      const aiInstance = new ConnectFourAI("/connect-four/alphazero-network-model.onnx");
      await aiInstance.init();
      setAi(aiInstance);
      setAiLoading(false);
      setAiInitialized(true);
      console.log('AI initialized successfully');
    } catch (error) {
      console.error('Failed to initialize AI:', error);
      setAiError(error.message);
      setAiLoading(false);
      
      // Auto-retry up to 3 times
      if (retryCount < 2) {
        console.log(`Retrying AI initialization in 2 seconds... (attempt ${retryCount + 2})`);
        setTimeout(() => {
          setAiRetryCount(retryCount + 1);
          initializeAI(retryCount + 1);
        }, 2000);
      }
    }
  };

  const retryAI = () => {
    setAiRetryCount(0);
    setAiInitialized(false);
    initializeAI(0);
  };

  // Check if AlphaZero is selected and AI needs to be initialized
  const needsAI = (playerTypes.red === 'ai-alphazero' || playerTypes.yellow === 'ai-alphazero') && !aiInitialized && !aiLoading;
  
  useEffect(() => {
    if (needsAI) {
      initializeAI();
    }
  }, [needsAI]);

  const handlePlayersChanged = (types) => {
    setPlayerTypes(types);
  };

  return (
    <div className="app">
      <h1>Connect Four</h1>
      
      {/* AI Loading/Error Status */}
      {aiLoading && (
        <div className="ai-status loading">
          <div className="loading-spinner"></div>
          <span>Loading AI model... {aiRetryCount > 0 && `(Attempt ${aiRetryCount + 1})`}</span>
        </div>
      )}
      
      {aiError && !aiLoading && (
        <div className="ai-status error">
          <div className="error-icon">⚠️</div>
          <div className="error-content">
            <p><strong>AI Loading Failed:</strong> {aiError}</p>
            <button onClick={retryAI} className="retry-button">
              Retry Loading AI
            </button>
          </div>
        </div>
      )}
      
      {needsAI && (
        <div className="ai-status warning">
          <div className="warning-icon">⚠️</div>
          <div className="warning-content">
            <p><strong>AI Required:</strong> Please wait for AI to load or select human players.</p>
            <button onClick={retryAI} className="retry-button">
              Retry Loading AI
            </button>
          </div>
        </div>
      )}
      
      {/* Block interface while AI is loading */}
      <div className={aiLoading ? 'interface-blocked' : ''}>
        <Board 
          playerTypes={playerTypes} 
          onPlayersChanged={handlePlayersChanged}
          ai={ai}
          aiLoading={aiLoading}
          aiError={aiError}
          aiInitialized={aiInitialized}
        />
      </div>
    </div>
  );
}

export default App;
