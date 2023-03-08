package linear;

public class VectorF32 {
    final int size;
    float[] data;

    public VectorF32(int size, float[] data) {
        this.size = size;
        this.data = data;
    }

    public VectorF32(float[] data) {
        this.size = data.length;
        this.data = data;
    }

    public VectorF32(int size) {
        this.size = size;
        this.data = new float[size];
    }

    public int getSize() {
        return size;
    }

    public float[] getData() {
        return data;
    }

    public void setData(float[] data) {
        this.data = data;
    }

    public MatrixF32 toVerticalMatrix() {
        return new MatrixF32(this.getSize(), 1, this.getData());
    }

    public MatrixF32 toHorizontalMatrix() {
        return new MatrixF32(1, this.getSize(), this.getData());
    }
}
