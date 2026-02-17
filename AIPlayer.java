import java.util.ArrayList;
import java.util.Random;

public class AIPlayer extends Player {
    private Random random = new Random();

    public AIPlayer(char symbol, String name) {
        super(symbol, name);
    }

    @Override
    public int[] makeMove(Board board) {
        ArrayList<int[]> moves = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board.isValidMove(i, j)) {
                    moves.add(new int[]{i, j});
                }
            }
        }
        if (!moves.isEmpty()) {
            int[] move = moves.get(random.nextInt(moves.size()));
            System.out.println(this.name + " (" + symbol + ") chooses: " + move[0] + ", " + move[1]);
            return move;
        }
        return null; // Should not happen
    }
}
    

