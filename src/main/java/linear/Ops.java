package linear;

import dev.ludovic.netlib.BLAS;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class Ops {

    public static final VectorSpecies<Float> species = FloatVector.SPECIES_MAX;

    public static void assertNoNan(float[] d) {
//        for (var i : d) {
//            if (Float.isNaN(i)) {
//                throw new RuntimeException();
//            }
//        }
    }

    public static VectorF32 multipleElements(VectorF32 vector1, VectorF32 vector2, VectorF32 result, float alpha, float beta) {
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

    public static VectorF32 product(VectorF32 vector, MatrixF32 matrix, VectorF32 result, float alpha, float beta) {
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

    public static VectorF32 product(MatrixF32 matrix, VectorF32 vector, VectorF32 result, float alpha, float beta) {
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

    public static MatrixF32 product(VectorF32 vector1, VectorF32 vector2, MatrixF32 result, float alpha, float beta) {
        productF32Blas(result.getData(), vector1.toVerticalMatrix(), vector2.toHorizontalMatrix(), vector1.getData(), vector2.getData(), alpha, beta);

        return result;
    }
    public static VectorF32 product(MatrixF32 matrix, VectorF32 vector, float alpha, float beta) {
        var result = new VectorF32(matrix.getRows());

        return product(matrix, vector, result, alpha, beta);
    }
    public static VectorF32 product(VectorF32 vector, MatrixF32 matrix, float alpha, float beta) {
        var result = new VectorF32(matrix.getColumns());

        return product(vector, matrix, result, alpha, beta);
    }
    public static MatrixF32 product(VectorF32 vector1, VectorF32 vector2, float alpha, float beta) {
        var result = new MatrixF32(vector1.getSize(), vector2.getSize());

        return product(vector1, vector2, result, alpha, beta);
    }

    public static VectorF32 product(MatrixF32 matrix, VectorF32 vector, float alpha) {
        return product(matrix, vector, alpha, 0.0f);
    }
    public static VectorF32 product(VectorF32 vector, MatrixF32 matrix, float alpha) {
        return product(vector, matrix, alpha, 0.0f);
    }
    public static MatrixF32 product(VectorF32 vector1, VectorF32 vector2, float alpha) {
        return product(vector1, vector2, alpha, 0.0f);
    }

    public static VectorF32 product(MatrixF32 matrix, VectorF32 vector) {
        return product(matrix, vector, 1.0f, 0.0f);
    }
    public static VectorF32 product(VectorF32 vector, MatrixF32 matrix) {
        return product(vector, matrix, 1.0f, 0.0f);
    }
    public static MatrixF32 product(VectorF32 vector1, VectorF32 vector2) {
        return product(vector1, vector2, 1.0f, 0.0f);
    }

    public static MatrixF32 product(MatrixF32 matrix1, MatrixF32 matrix2) {
        return product(matrix1, matrix2, 1.0f, 0.0f);
    }

    public static MatrixF32 product(MatrixF32 matrix1, MatrixF32 matrix2, float alpha, float beta) {
        var result = new MatrixF32(matrix1.getRows(), matrix2.getColumns(), new float[matrix1.getRows() * matrix2.getColumns()]);

        return product(matrix1, matrix2, result, alpha, beta);
    }

    public static MatrixF32 product(MatrixF32 matrix1, MatrixF32 matrix2, MatrixF32 result, float alpha, float beta) {
        if (matrix1.getColumns() != matrix2.getRows()) {
            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        productF32Blas(resultData, matrix1, matrix2, data1, data2, alpha, beta);

        return result;
    }

    public static MatrixF32 product(MatrixF32 matrix1, float alpha) {
        getBlas().sscal(matrix1.getSize(), alpha, matrix1.getData(), 1);

        return matrix1;
    }

    public static void add(MatrixF32 x, MatrixF32 y, float alpha) {
        getBlas().saxpy(x.getSize(), alpha, x.getData(), 1, y.getData(), 1);
    }

    public static void add(VectorF32 x, VectorF32 y, float alpha) {
        getBlas().saxpy(x.getSize(), alpha, x.getData(), 1, y.getData(), 1);
    }

    public static float[] add(float[] x, float[] y, float alpha) {
        getBlas().saxpy(x.length, alpha, x, 1, y, 1);

        return y;
    }

    public static float amax(float[] x) {
        var i = getBlas().isamax(x.length, x, 1);

        return x[i];
    }

    public static float max(float[] x, int offset, int length) {
        var r = -Float.MAX_VALUE;
        for (var i = offset; i < offset + length; i++) {
            r = Math.max(r, x[i]);
        }

        return r;
    }

    private static void productF32Blas(float[] resultData, MatrixF32 matrix1, MatrixF32 matrix2, float[] data1, float[] data2, float alpha, float beta) {
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
    }
}
