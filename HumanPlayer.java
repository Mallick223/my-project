import java.util.Scanner;

public class HumanPlayer extends Player {
    public HumanPlayer(char symbol, String name) {
        super(symbol, name);
    }

    @Override
    public int[] makeMove(Board board) {
        Scanner scanner = new Scanner(System.in);
        int row, col;
        while (true) {
            System.out.print(this.name + " (" + symbol + "), enter row (0-2) and col (0-2): ");
            row = scanner.nextInt();
            col = scanner.nextInt();
            if (board.isValidMove(row, col)) {
                return new int[]{row, col};
            } else {
                System.out.println("Invalid move! Try again.");
            }
        }
    }
}
