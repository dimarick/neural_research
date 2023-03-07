package linear.matrix;

import net.dedekind.blas.Blas;

public class Ops {
    public static MatrixF32 multiple(MatrixF32 matrix1, MatrixF32 matrix2) {
        return multiple(matrix1, matrix2, 1.0f, 0.0f);
    }

    public static MatrixF32 multiple(MatrixF32 matrix1, MatrixF32 matrix2, float alpha, float beta) {
        var result = new MatrixF32(matrix1.getRows(), matrix2.getColumns(), new float[matrix1.getRows() * matrix2.getColumns()]);

        return multiple(matrix1, matrix2, result, alpha, beta);
    }

    public static MatrixF32 multiple(MatrixF32 matrix1, MatrixF32 matrix2, MatrixF32 result, float alpha, float beta) {
        if (matrix1.getColumns() != matrix2.getRows()) {
            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        multipleF32RangeBlas(resultData, matrix1, matrix2, data1, data2, alpha, beta);

        return result;
    }

    public static MatrixF32 transposeVector(MatrixF32 matrix) {
        if (matrix.getRows() > 1 && matrix.getColumns() > 1) {
            throw new RuntimeException("cannot transpose matrix");
        }

        return new MatrixF32(matrix.getColumns(), matrix.getRows(), matrix.getData());
    }

    public static float[] subtract(float[] a, float[] b) {
        var r = new float[a.length];

        for (var i = 0; i < a.length; i++) {
            r[i] = a[i] - b[i];
        }

        return r;
    }

    private static void multipleF32RangeBlas(float[] resultData, MatrixF32 matrix1, MatrixF32 matrix2, float[] data1, float[] data2, float alpha, float beta) {
        Blas.getInstance(true).sgemm("N", "N",
                matrix2.getColumns(), matrix1.getRows(),
                matrix2.getRows(), alpha, data2, 0, matrix2.getColumns(),
                data1, 0, matrix1.getColumns(),
                beta,
                resultData, 0, matrix2.getColumns());
    }
}
