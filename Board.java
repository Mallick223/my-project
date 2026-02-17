import java.io.*;
import java.util.Scanner;

public class Board {
    private char[][] grid = new char[3][3];

    public Board() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                grid[i][j] = ' ';
            }
        }
    }

    public boolean isValidMove(int row, int col) {
        return row >= 0 && row < 3 && col >= 0 && col < 3 && grid[row][col] == ' ';
    }

    public boolean makeMove(int row, int col, char symbol) {
        if (isValidMove(row, col)) {
            grid[row][col] = symbol;
            return true;
        }
        return false;
    }

    public boolean checkWin(char symbol) {
        // Rows, columns, diagonals
        for (int i = 0; i < 3; i++) {
            if (grid[i][0] == symbol && grid[i][1] == symbol && grid[i][2] == symbol) return true;
            if (grid[0][i] == symbol && grid[1][i] == symbol && grid[2][i] == symbol) return true;
        }
        if (grid[0][0] == symbol && grid[1][1] == symbol && grid[2][2] == symbol) return true;
        if (grid[0][2] == symbol && grid[1][1] == symbol && grid[2][0] == symbol) return true;
        return false;
    }

    public boolean isFull() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (grid[i][j] == ' ') return false;
            }
        }
        return true;
    }

    public void printBoard() {
        System.out.println("  0 1 2");
        for (int i = 0; i < 3; i++) {
            System.out.print(i + " ");
            for (int j = 0; j < 3; j++) {
                System.out.print(grid[i][j] + " | ");
            }
            System.out.println();
            if (i < 2) System.out.println("  ---------");
        }
        System.out.println();
    }

    // Save board to file
    public void saveToFile(String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    writer.print(grid[i][j]);
                    if (j < 2) writer.print(" ");
                }
                writer.println(); // New line for each row
            }
            System.out.println("Game saved to " + filename);
        } catch (IOException e) {
            System.out.println("Error saving game: " + e.getMessage());
        }
    }

    // Load board from file
    public boolean loadFromFile(String filename) {
        try (Scanner scanner = new Scanner(new File(filename))) {
            for (int i = 0; i < 3; i++) {
                if (scanner.hasNextLine()) {
                    String line = scanner.nextLine().trim();
                    if (line.length() != 5) { // Expect "X O " (3 chars + 2 spaces)
                        throw new IOException("Invalid file format");
                    }
                    for (int j = 0; j < 3; j++) {
                        int pos = j * 2; // Positions: 0,2,4
                        grid[i][j] = line.charAt(pos);
                    }
                } else {
                    throw new IOException("File too short");
                }
            }
            System.out.println("Game loaded from " + filename);
            return true;
        } catch (FileNotFoundException e) {
            System.out.println("Save file not found: " + filename + ". Starting new game.");
            return false;
        } catch (IOException e) {
            System.out.println("Error loading game: " + e.getMessage() + ". Starting new game.");
            return false;
        }
    }

    // NEW: Helper methods for encapsulation (used by GameEngine)
    public int getXCount() {
        int count = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (grid[i][j] == 'X') count++;
            }
        }
        return count;
    }

    public int getOCount() {
        int count = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (grid[i][j] == 'O') count++;
            }
        }
        return count;
    }
}