class TicTacToe:
    """Original TicTacToe environment from notebook."""
    def __init__(self, size=3):
        self.size = size
        self.board = [' ' for _ in range(size*size)]
        self.current_winner = None

    def print_board(self):
        for row in [self.board[i * self.size:(i + 1) * self.size] for i in range(self.size)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums(size):
        number_board = [[str(i) for i in range(j * size, (j + 1) * size)] for j in range(size)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def total_moves(self):
        return list(range(self.size*self.size))

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // self.size
        row = self.board[row_ind * self.size:(row_ind + 1) * self.size]
        if all([spot == letter for spot in row]):
            return True
        col_ind = square % self.size
        column = [self.board[col_ind + i * self.size] for i in range(self.size)]
        if all([spot == letter for spot in column]):
            return True
        if square % (self.size + 1) == 0:
            diagonal1 = [self.board[i] for i in range(0, self.size * self.size, self.size + 1)]
            if all([spot == letter for spot in diagonal1]):
                return True
        if square % (self.size - 1) == 0 and square not in (0, self.size*self.size - 1):
            diagonal2 = [self.board[i] for i in range(self.size - 1, self.size * self.size - 1, self.size - 1)]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False
