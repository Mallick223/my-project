public class GameEngine {
    private Board board;
    private Player player1;
    private Player player2;
    private Player currentPlayer;
    private static final String SAVE_FILE = "game_save.txt";

    public GameEngine(Player p1, Player p2) {
        this.board = new Board();
        this.player1 = p1;
        this.player2 = p2;
        this.currentPlayer = player1;
    }

    public GameEngine(Player p1, Player p2, Board loadedBoard) {
        this.board = loadedBoard;
        this.player1 = p1;
        this.player2 = p2;
        // Determine current player: If X count == O count, X's turn; else O's turn
        int xCount = board.getXCount();
        int oCount = board.getOCount();
        this.currentPlayer = (xCount == oCount) ? player1 : player2;
    }

    public void play() {
        System.out.println("Tic-Tac-Toe Game Started!");
        while (true) {
            board.printBoard();
            int[] move = currentPlayer.makeMove(board);
            if (board.makeMove(move[0], move[1], currentPlayer.getSymbol())) {
                // Auto-save after each move
                board.saveToFile(SAVE_FILE);
                if (board.checkWin(currentPlayer.getSymbol())) {
                    board.printBoard();
                    System.out.println(currentPlayer.getName() + " wins!");
                    break;
                } else if (board.isFull()) {
                    board.printBoard();
                    System.out.println("It's a draw!");
                    break;
                }
            } else {
                System.out.println("Invalid move—skipping turn.");
            }
            currentPlayer = (currentPlayer == player1) ? player2 : player1;
        }
        // Final save
        board.saveToFile(SAVE_FILE);
    }
}