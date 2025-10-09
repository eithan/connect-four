export class ConnectFourGame {
  constructor() {
    this.reset();
  }

  reset() {
    this.board = Array(6).fill().map(() => Array(7).fill(null));
    this.currentPlayer = 'red';
    this.winner = null;
    this.winningCells = [];
  }

  getState() {
    return {
      board: this.board,
      currentPlayer: this.currentPlayer,
      winner: this.winner,
      winningCells: this.winningCells
    };
  }

  makeMove(column) {
    if (this.winner) return false;
    
    const row = this.findLowestEmptyRow(column);
    if (row === -1) return false;

    this.board[row][column] = this.currentPlayer;
    
    if (this.checkWin(row, column)) {
      this.winner = this.currentPlayer;
    } else if (this.isBoardFull()) {
      this.winner = 'draw';
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

  // Make this method accessible for checking available moves
  getAvailableColumns() {
    const availableColumns = [];
    for (let col = 0; col < 7; col++) {
      if (this.findLowestEmptyRow(col) !== -1) {
        availableColumns.push(col);
      }
    }
    return availableColumns;
  }

  isBoardFull() {
    // Check if all columns are full (top row has no empty spaces)
    for (let col = 0; col < 7; col++) {
      if (this.board[0][col] === null || this.board[0][col] === undefined) {
        return false;
      }
    }
    return true;
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
      const cells = [{row, column}];
      
      // Count in positive direction
      const positiveCells = this.countDirectionWithCells(row, column, dr, dc);
      count += positiveCells.count;
      cells.push(...positiveCells.cells);
      
      // Count in negative direction
      const negativeCells = this.countDirectionWithCells(row, column, -dr, -dc);
      count += negativeCells.count;
      cells.push(...negativeCells.cells);
      
      if (count >= 4) {
        this.winningCells = cells;
        return true;
      }
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

  countDirectionWithCells(row, column, dr, dc) {
    const player = this.board[row][column];
    let count = 0;
    const cells = [];
    let r = row + dr;
    let c = column + dc;

    while (
      r >= 0 && r < 6 && 
      c >= 0 && c < 7 && 
      this.board[r][c] === player
    ) {
      count++;
      cells.push({row: r, column: c});
      r += dr;
      c += dc;
    }
    return { count, cells };
  }
} 