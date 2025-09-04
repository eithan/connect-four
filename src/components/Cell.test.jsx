/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { jest } from '@jest/globals';
import Cell from './Cell';

describe('Cell Component', () => {
  test('renders empty cell', () => {
    const { container } = render(<Cell value={null} onClick={() => {}} />);
    expect(container.firstChild).toHaveClass('cell');
    expect(container.querySelector('.piece')).toBeNull();
  });

  test('renders red piece when value is 1', () => {
    const { container } = render(<Cell value={1} onClick={() => {}} />);
    const piece = container.querySelector('.piece');
    expect(piece).toHaveClass('red');
  });

  test('renders yellow piece when value is 2', () => {
    const { container } = render(<Cell value={2} onClick={() => {}} />);
    const piece = container.querySelector('.piece');
    expect(piece).toHaveClass('yellow');
  });

  test('calls onClick when cell is clicked', () => {
    const handleClick = jest.fn();
    const { container } = render(<Cell value={null} onClick={handleClick} />);
    fireEvent.click(container.firstChild);
    expect(handleClick).toHaveBeenCalled();
  });
}); 