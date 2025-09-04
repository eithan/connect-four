import { ConnectFourGame } from './connectFour';

describe('ConnectFourGame', () => {
  let game;

  beforeEach(() => {
    game = new ConnectFourGame();
  });

  describe('initial state', () => {
    test('starts with empty board', () => {
      const { board } = game.getState();
      expect(board.every(row => row.every(cell => cell === null))).toBe(true);
    });

    test('starts with red player', () => {
      const { currentPlayer } = game.getState();
      expect(currentPlayer).toBe('red');
    });

    test('starts with no winner', () => {
      const { winner } = game.getState();
      expect(winner).toBeNull();
    });
  });

  describe('makeMove', () => {
    test('places piece at bottom of empty column', () => {
      game.makeMove(0);
      const { board } = game.getState();
      expect(board[5][0]).toBe("red"); // Red piece at bottom
    });

    test('stacks pieces in same column', () => {
      game.makeMove(0); // Red at bottom
      game.makeMove(0); // Yellow above
      const { board } = game.getState();
      expect(board[5][0]).toBe("red");
      expect(board[4][0]).toBe("yellow");
    });

    test('alternates between players', () => {
      game.makeMove(0);
      expect(game.getState().currentPlayer).toBe('yellow');
      game.makeMove(1);
      expect(game.getState().currentPlayer).toBe('red');
    });

    test('prevents moves in full column', () => {
      // Fill column 0
      for (let i = 0; i < 6; i++) {
        game.makeMove(0);
      }
      expect(game.makeMove(0)).toBe(false);
    });

    test('prevents moves after game is won', () => {
      // Create horizontal win for red
      game.makeMove(0); // Red
      game.makeMove(0); // Yellow
      game.makeMove(1); // Red
      game.makeMove(1); // Yellow
      game.makeMove(2); // Red
      game.makeMove(2); // Yellow
      game.makeMove(3); // Red wins
      
      expect(game.makeMove(4)).toBe(false);
    });
  });

  describe('win detection', () => {
    test('detects horizontal win', () => {
      // Red plays winning horizontal line
      game.makeMove(0); // Red
      game.makeMove(0); // Yellow
      game.makeMove(1); // Red
      game.makeMove(1); // Yellow
      game.makeMove(2); // Red
      game.makeMove(2); // Yellow
      game.makeMove(3); // Red wins
      
      expect(game.getState().winner).toBe('red');
    });

    test('detects vertical win', () => {
      // Red plays winning vertical line
      game.makeMove(0); // Red
      game.makeMove(1); // Yellow
      game.makeMove(0); // Red
      game.makeMove(1); // Yellow
      game.makeMove(0); // Red
      game.makeMove(1); // Yellow
      game.makeMove(0); // Red wins
      
      expect(game.getState().winner).toBe('red');
    });

    test('detects diagonal win (up-right)', () => {
      // Create diagonal win for red
      game.makeMove(0); // Red
      game.makeMove(1); // Yellow
      game.makeMove(1); // Red
      game.makeMove(2); // Yellow
      game.makeMove(2); // Red
      game.makeMove(3); // Yellow
      game.makeMove(2); // Red
      game.makeMove(3); // Yellow
      game.makeMove(3); // Red
      game.makeMove(0); // Yellow
      game.makeMove(3); // Red wins
      
      expect(game.getState().winner).toBe('red');
    });

    test('detects diagonal win (up-left)', () => {
      // Create diagonal win for red
      game.makeMove(3); // Red
      game.makeMove(2); // Yellow
      game.makeMove(2); // Red
      game.makeMove(1); // Yellow
      game.makeMove(1); // Red
      game.makeMove(0); // Yellow
      game.makeMove(1); // Red
      game.makeMove(0); // Yellow
      game.makeMove(0); // Red
      game.makeMove(3); // Yellow
      game.makeMove(0); // Red wins
      
      expect(game.getState().winner).toBe('red');
    });
  });

  describe('reset', () => {
    test('clears board', () => {
      game.makeMove(0);
      game.makeMove(1);
      game.reset();
      const { board } = game.getState();
      expect(board.every(row => row.every(cell => cell === null))).toBe(true);
    });

    test('resets to red player', () => {
      game.makeMove(0);
      game.reset();
      expect(game.getState().currentPlayer).toBe('red');
    });

    test('clears winner', () => {
      // Create win then reset
      game.makeMove(0);
      game.makeMove(1);
      game.makeMove(0);
      game.makeMove(1);
      game.makeMove(0);
      game.makeMove(1);
      game.makeMove(0); // Red wins
      game.reset();
      expect(game.getState().winner).toBeNull();
    });
  });
}); 