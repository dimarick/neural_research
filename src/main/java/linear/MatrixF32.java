package linear;

public class MatrixF32 {
    private  final int rows;
    private final int columns;
    private boolean transposed = false;
    float[] data;

    public MatrixF32(int rows, int columns, float[] data, boolean transposed) {
        this.rows = rows;
        this.columns = columns;
        this.data = data;
        this.transposed = transposed;

        if (data.length != rows * columns) {
            throw new ArrayIndexOutOfBoundsException("data length is invalid");
        }
    }

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

    public int getSize() {
        return rows * columns;
    }

    public boolean isTransposed() {
        return transposed;
    }

    public float[] getData() {
        return data;
    }
//
//    public void setData(float[] data) {
//        this.data = data;
//    }

    public MatrixF32 transpose() {
        return new MatrixF32(this.columns, this.rows, data, !transposed);
    }
}
