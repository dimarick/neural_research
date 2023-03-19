package linear;

import dev.ludovic.netlib.BLAS;

public class Ops {
    public static VectorF32 multipleBand(VectorF32 vector1, VectorF32 vector2, VectorF32 result, float alpha, float beta) {
        getBlas().ssbmv("L",
                vector1.getSize(),
                0,
                alpha,
                vector1.getData(),
                1,
                vector2.getData(),
                1,
                beta,
                result.getData(),
                1
        );

        return result;
    }

    public static VectorF32 multiple(VectorF32 vector, MatrixF32 matrix, VectorF32 result, float alpha, float beta) {
        getBlas().sgemv(
                matrix.isTransposed() ? "T" : "N",
                matrix.isTransposed() ? matrix.getRows() : matrix.getColumns(),
                matrix.isTransposed() ? matrix.getColumns() : matrix.getRows(),
                alpha,
                matrix.getData(),
                matrix.isTransposed() ? matrix.getRows() : matrix.getColumns(),
                vector.getData(),
                1,
                beta,
                result.getData(),
                1
        );

        return result;
    }

    public static VectorF32 multiple(MatrixF32 matrix, VectorF32 vector, VectorF32 result, float alpha, float beta) {
        getBlas().sgemv(
                matrix.isTransposed() ? "T" : "N",
                matrix.isTransposed() ? matrix.getColumns() : matrix.getRows(),
                matrix.isTransposed() ? matrix.getRows() : matrix.getColumns(),
                alpha,
                matrix.getData(),
                matrix.isTransposed() ? matrix.getColumns() : matrix.getRows(),
                vector.getData(),
                1,
                beta,
                result.getData(),
                1
        );

        return result;
    }

    private static BLAS getBlas() {
        return BLAS.getInstance();
    }

    public static MatrixF32 multiple(VectorF32 vector1, VectorF32 vector2, MatrixF32 result, float alpha, float beta) {
        multipleF32Blas(result.getData(), vector1.toVerticalMatrix(), vector2.toHorizontalMatrix(), vector1.getData(), vector2.getData(), alpha, beta);

        return result;
    }
    public static VectorF32 multiple(MatrixF32 matrix, VectorF32 vector, float alpha, float beta) {
        var result = new VectorF32(matrix.getRows());

        return multiple(matrix, vector, result, alpha, beta);
    }
    public static VectorF32 multiple(VectorF32 vector, MatrixF32 matrix, float alpha, float beta) {
        var result = new VectorF32(matrix.getColumns());

        return multiple(vector, matrix, result, alpha, beta);
    }
    public static MatrixF32 multiple(VectorF32 vector1, VectorF32 vector2, float alpha, float beta) {
        var result = new MatrixF32(vector1.getSize(), vector2.getSize());

        return multiple(vector1, vector2, result, alpha, beta);
    }

    public static VectorF32 multiple(MatrixF32 matrix, VectorF32 vector, float alpha) {
        return multiple(matrix, vector, alpha, 0.0f);
    }
    public static VectorF32 multiple(VectorF32 vector, MatrixF32 matrix, float alpha) {
        return multiple(vector, matrix, alpha, 0.0f);
    }
    public static MatrixF32 multiple(VectorF32 vector1, VectorF32 vector2, float alpha) {
        return multiple(vector1, vector2, alpha, 0.0f);
    }

    public static VectorF32 multiple(MatrixF32 matrix, VectorF32 vector) {
        return multiple(matrix, vector, 1.0f, 0.0f);
    }
    public static VectorF32 multiple(VectorF32 vector, MatrixF32 matrix) {
        return multiple(vector, matrix, 1.0f, 0.0f);
    }
    public static MatrixF32 multiple(VectorF32 vector1, VectorF32 vector2) {
        return multiple(vector1, vector2, 1.0f, 0.0f);
    }

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

        multipleF32Blas(resultData, matrix1, matrix2, data1, data2, alpha, beta);

        return result;
    }

    public static float[] add(float[] x, float[] y, float alpha) {
        getBlas().saxpy(x.length, alpha, x, 1, y, 1);

        return y;
    }

    public static void copy(float[] x, float[] y) {
        getBlas().scopy(x.length, x, 1, y, 1);
    }

    private static void multipleF32Blas(float[] resultData, MatrixF32 matrix1, MatrixF32 matrix2, float[] data1, float[] data2, float alpha, float beta) {
        getBlas().sgemm(
                matrix2.isTransposed() ? "T" : "N",
                matrix1.isTransposed() ? "T" : "N",
                matrix2.getColumns(),
                matrix1.getRows(),
                matrix2.getRows(),
                alpha,
                data2,
                0,
                matrix2.isTransposed() ? matrix2.getRows() : matrix2.getColumns(),
                data1,
                0,
                matrix1.isTransposed() ? matrix1.getRows() : matrix1.getColumns(),
                beta,
                resultData,
                0,
                matrix2.getColumns());
//        getBlas().sgemm(
//                matrix1.isTransposed() ? "T" : "N",
//                matrix2.isTransposed() ? "T" : "N",
//                matrix2.isTransposed() ? matrix2.getRows() : matrix2.getColumns(),
//                matrix1.isTransposed() ? matrix1.getColumns() : matrix1.getRows(),
//                matrix2.isTransposed() ? matrix2.getColumns() : matrix2.getRows(),
//                alpha,
//                data2,
//                0,
//                matrix2.isTransposed() ? matrix2.getRows() : matrix2.getColumns(),
//                data1,
//                0,
//                matrix1.isTransposed() ? matrix1.getRows() : matrix1.getColumns(),
//                beta,
//                resultData,
//                0,
//                matrix2.isTransposed() ? matrix2.getRows() : matrix2.getColumns());
    }
}
