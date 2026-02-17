import java.util.Scanner;

public class TicTacToeApp {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== Tic-Tac-Toe with Save/Load ===");
        System.out.println("1. New Game");
        System.out.println("2. Load Saved Game");
        System.out.print("Choose (1 or 2): ");
        int choice = scanner.nextInt();

        Player human = new HumanPlayer('X', "Human");
        Player ai = new AIPlayer('O', "AI Bot");
        GameEngine game;

        if (choice == 2) {
            Board loadedBoard = new Board();
            if (loadedBoard.loadFromFile("game_save.txt")) {
                game = new GameEngine(human, ai, loadedBoard);
                System.out.println("Resuming from saved state...");
            } else {
                game = new GameEngine(human, ai);
                System.out.println("Could not load—starting new game.");
            }
        } else {
            game = new GameEngine(human, ai);
        }

        game.play();
        scanner.close();
    }
}