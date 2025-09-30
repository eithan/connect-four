import React from 'react';

const PlayerMenu = ({ onPlayersSelected }) => {
  const [redType, setRedType] = React.useState('human');
  const [yellowType, setYellowType] = React.useState('human');

  const handleSubmit = (e) => {
    e.preventDefault();
    onPlayersSelected({ red: redType, yellow: yellowType });
  };

  return (
    <form onSubmit={handleSubmit} className="player-menu">
      <div className="player-select">
        <label>
          <span className="player-label red">Red Player</span>
          <select value={redType} onChange={(e) => setRedType(e.target.value)}>
            <option value="human">Human</option>
            <option value="computer">Computer</option>
          </select>
        </label>
      </div>
      <div className="player-select">
        <label>
          <span className="player-label yellow">Yellow Player</span>
          <select value={yellowType} onChange={(e) => setYellowType(e.target.value)}>
            <option value="human">Human</option>
            <option value="computer">Computer</option>
          </select>
        </label>
      </div>
      <button type="submit">Start Game</button>
    </form>
  );
};

export default PlayerMenu; 