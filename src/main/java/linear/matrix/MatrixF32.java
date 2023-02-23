package linear.matrix;

public class MatrixF32 implements MatrixF32Interface {
    final int rows;
    final public int columns;
    float[] data;

    public MatrixF32(int rows, int columns, float[] data) {
        this.rows = rows;
        this.columns = columns;
        this.data = data;

        if (data.length != rows * columns) {
            throw new ArrayIndexOutOfBoundsException("data length is invalid");
        }
    }

    public MatrixF32(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = new float[rows * columns];
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public float[] getData() {
        return data;
    }

    public void setData(float[] data) {
        this.data = data;
    }
}
