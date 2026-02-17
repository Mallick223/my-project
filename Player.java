public abstract class Player {
    protected char symbol;
    protected String name;

    public Player(char symbol, String name) {
        this.symbol = symbol;
        this.name = name;
    }

    public char getSymbol() { return symbol; }
    public String getName() { return name; }

    // Abstract method for polymorphism: overridden by subclasses
    public abstract int[] makeMove(Board board);
}