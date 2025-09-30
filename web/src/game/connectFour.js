export class ConnectFourGame {
  constructor() {
    this.reset();
  }

  reset() {
    this.board = Array(6).fill().map(() => Array(7).fill(null));
    this.currentPlayer = 'red';
    this.winner = null;
  }

  getState() {
    return {
      board: this.board,
      currentPlayer: this.currentPlayer,
      winner: this.winner
    };
  }

  makeMove(column) {
    if (this.winner) return false;
    
    const row = this.findLowestEmptyRow(column);
    if (row === -1) return false;

    this.board[row][column] = this.currentPlayer;
    
    if (this.checkWin(row, column)) {
      this.winner = this.currentPlayer;
    } else {
      this.currentPlayer = this.currentPlayer === 'red' ? 'yellow' : 'red';
    }

    return { row, column };
  }

  findLowestEmptyRow(column) {
    for (let row = 5; row >= 0; row--) {
      if (!this.board[row][column]) return row;
    }
    return -1;
  }

  checkWin(row, column) {
    const directions = [
      [0, 1],   // horizontal
      [1, 0],   // vertical
      [1, 1],   // diagonal up-right
      [1, -1],  // diagonal up-left
    ];

    for (const [dr, dc] of directions) {
      let count = 1;
      count += this.countDirection(row, column, dr, dc);
      count += this.countDirection(row, column, -dr, -dc);
      if (count >= 4) return true;
    }
    return false;
  }

  countDirection(row, column, dr, dc) {
    const player = this.board[row][column];
    let count = 0;
    let r = row + dr;
    let c = column + dc;

    while (
      r >= 0 && r < 6 && 
      c >= 0 && c < 7 && 
      this.board[r][c] === player
    ) {
      count++;
      r += dr;
      c += dc;
    }
    return count;
  }
} 