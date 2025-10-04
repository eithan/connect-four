import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import Board from './Board';

describe('Board Component', () => {
  let component;

  beforeEach(() => {
    component = render(<Board />);
  });

  test('renders empty board initially', () => {
    const cells = component.container.querySelectorAll('.cell');
    expect(cells.length).toBe(42); // 7x6 board
    expect(component.container.querySelectorAll('.piece').length).toBe(0);
  });

  test('shows current player status', () => {
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const status = component.getByTestId('game-status');
    expect(status.textContent).toContain("Red's turn");
  });

  test('adds piece when cell is clicked', () => {
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const cells = component.container.querySelectorAll('.cell');
    fireEvent.click(cells[35]); // Bottom row, first column
    expect(component.container.querySelectorAll('.piece').length).toBe(1);
  });

  test('alternates between players', () => {
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const cells = component.container.querySelectorAll('.cell');
    const status = component.getByTestId('game-status');
    
    fireEvent.click(cells[35]); // First move
    expect(status.textContent).toContain("Yellow's turn");
    
    fireEvent.click(cells[34]); // Second move
    expect(status.textContent).toContain("Red's turn");
  });

  test('resets game when reset button is clicked', () => {
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const cells = component.container.querySelectorAll('.cell');
    fireEvent.click(cells[35]);
    expect(component.container.querySelectorAll('.piece').length).toBe(1);
    
    const resetButton = component.getByText('Reset Game');
    fireEvent.click(resetButton);
    expect(component.container.querySelectorAll('.piece').length).toBe(0);
  });

  test('shows AI loading status when AI is loading', () => {
    const playerTypes = { red: 'ai-alphazero', yellow: 'human' };
    const aiLoading = true;
    const aiError = null;
    const aiInitialized = false;
    
    component.rerender(
      <Board 
        playerTypes={playerTypes} 
        ai={null} 
        aiLoading={aiLoading} 
        aiError={aiError}
        aiInitialized={aiInitialized}
      />
    );
    
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const status = component.getByTestId('game-status');
    expect(status.textContent).toContain('Red (AI-AlphaZero) is loading...');
  });

  test('shows AI error status when AI has error', () => {
    const playerTypes = { red: 'ai-alphazero', yellow: 'human' };
    const aiLoading = false;
    const aiError = 'Failed to load model';
    const aiInitialized = false;
    
    component.rerender(
      <Board 
        playerTypes={playerTypes} 
        ai={null} 
        aiLoading={aiLoading} 
        aiError={aiError}
        aiInitialized={aiInitialized}
      />
    );
    
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const status = component.getByTestId('game-status');
    expect(status.textContent).toContain('Red (AI-AlphaZero) - Error: Using random moves');
  });

  test('shows normal AI status when AI is loaded', () => {
    const playerTypes = { red: 'ai-alphazero', yellow: 'human' };
    const aiLoading = false;
    const aiError = null;
    const aiInitialized = true;
    const mockAI = { getMove: () => Promise.resolve(0) };
    
    component.rerender(
      <Board 
        playerTypes={playerTypes} 
        ai={mockAI} 
        aiLoading={aiLoading} 
        aiError={aiError}
        aiInitialized={aiInitialized}
      />
    );
    
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const status = component.getByTestId('game-status');
    expect(status.textContent).toContain("Red (AI-AlphaZero)'s turn");
  });

  test('shows initializing status when AI is not initialized', () => {
    const playerTypes = { red: 'ai-alphazero', yellow: 'human' };
    const aiLoading = false;
    const aiError = null;
    const aiInitialized = false;
    
    component.rerender(
      <Board 
        playerTypes={playerTypes} 
        ai={null} 
        aiLoading={aiLoading} 
        aiError={aiError}
        aiInitialized={aiInitialized}
      />
    );
    
    const startButton = component.getByText('Start Game');
    fireEvent.click(startButton);
    
    const status = component.getByTestId('game-status');
    expect(status.textContent).toContain('Red (AI-AlphaZero) - Initializing...');
  });
}); 