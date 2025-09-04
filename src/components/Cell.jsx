import React from 'react';
import './Cell.css';

const Cell = ({ value, onClick, animate }) => {
  return (
    <div className="cell" onClick={onClick}>
      {value !== null && (
        <div className={`piece ${value === 1 ? 'red' : 'yellow'} ${animate ? 'drop-animation' : ''}`} />
      )}
    </div>
  );
};

export default Cell; 