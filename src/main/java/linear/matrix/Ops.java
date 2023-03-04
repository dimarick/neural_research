package linear.matrix;

import net.dedekind.blas.Blas;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Ops {

    final private static ArrayList<Thread> threads = new ArrayList<>();

    final private static ArrayList<ThreadItem> threadItems = new ArrayList<>();

    /**
     * Управление уже созданным пулом потоков намного более эффективно чем их создание/уничтожение
     */
    private static ArrayList<ThreadItem> initThreads(int maxThreads) {
        int append = maxThreads - threads.size();
        for (var i = 0; i < append; i++) {
            var threadItem = new ThreadItem();
            var t = new Thread(threadItem);
            t.start();
            threadItem.prepare();
            threads.add(t);
            threadItems.add(threadItem);
        }

        return threadItems;
    }

    public static MatrixF32Interface multiple(MatrixF32Interface matrix1, MatrixF32Interface matrix2) {
        return multiple(matrix1, matrix2, 1.0f, 0.0f);
    }

    public static MatrixF32Interface multiple(MatrixF32Interface matrix1, MatrixF32Interface matrix2, float alpha, float beta) {
        var result = new MatrixF32(matrix1.getRows(), matrix2.getColumns(), new float[matrix1.getRows() * matrix2.getColumns()]);

        return multiple(matrix1, matrix2, result, alpha, beta);
    }

    public static MatrixF32Interface multiple(MatrixF32Interface matrix1, MatrixF32Interface matrix2, MatrixF32Interface result, float alpha, float beta) {
        if (matrix1.getColumns() != matrix2.getRows()) {
            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        multipleF32RangeBlas(resultData, matrix1, matrix2, data1, data2, alpha, beta);

        return result;
    }

    /**
     * Умножить на транспонированную матрицу: это выгоднее прямого умножения за счет последовательного доступа к памяти и более эффективной работы с кешем процессора
     */
    public static MatrixF32Interface multipleTransposed(MatrixF32Interface matrix1, MatrixF32Interface matrix2) {
        if (matrix1.getColumns() != matrix2.getColumns()) {
            if (matrix2.getColumns() == 1 && matrix1.getColumns() == matrix2.getRows()) {
                return multipleTransposed(matrix1, transposeVector(matrix2));
            }

            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        var result = new MatrixF32(matrix1.getRows(), matrix2.getRows(), new float[matrix1.getRows() * matrix2.getRows()]);
        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        multipleTransposedF32Range(resultData, matrix1, matrix2, data1, data2, 0, matrix1.getRows());

        return result;
    }

    public static MatrixF32Interface transposeVector(MatrixF32Interface matrix) {
        if (matrix.getRows() > 1 && matrix.getColumns() > 1) {
            throw new RuntimeException("cannot transpose matrix");
        }

        return new MatrixF32(matrix.getColumns(), matrix.getRows(), matrix.getData());
    }

    public static MatrixF32Interface multipleConcurrent(MatrixF32Interface matrix1, MatrixF32Interface matrix2, int maxThreads) {
        if (maxThreads == 1) {
            return multiple(matrix1, matrix2);
        }

        if (matrix1.getColumns() != matrix2.getRows()) {
            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        var result = new MatrixF32(matrix1.getRows(), matrix2.getColumns(), new float[matrix1.getRows() * matrix2.getColumns()]);
        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        var threadItems = initThreads(maxThreads);

        var rowsPerThread = (int)Math.ceil((float)matrix1.getRows() / maxThreads);

        var actualThreadItems = new ArrayList<ThreadItem>();

        var threadNo = 0;
        for (var i = 0; i < matrix1.getRows(); i += rowsPerThread) {
            final var rangeStart = i;
            final var rangeLimit = Math.min(i + rowsPerThread, matrix1.getRows());
            var threadItem = threadItems.get(threadNo++);
            threadItem.setFn(() -> multipleF32Range(resultData, matrix1, matrix2, data1, data2, rangeStart, rangeLimit));
            actualThreadItems.add(threadItem);
        }

        for (var t : actualThreadItems) {
            t.waitTask();
        }

        return result;
    }

    public static MatrixF32Interface multipleTransposedConcurrent(MatrixF32Interface matrix1, MatrixF32Interface matrix2, int maxThreads) {
        if (maxThreads == 1) {
            return multipleTransposed(matrix1, matrix2);
        }

        if (matrix1.getColumns() != matrix2.getColumns()) {
            if (matrix2.getColumns() == 1 && matrix1.getColumns() == matrix2.getRows()) {
                return multipleTransposedConcurrent(matrix1, transposeVector(matrix2), maxThreads);
            }

            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        var result = new MatrixF32(matrix1.getRows(), matrix2.getRows(), new float[matrix1.getRows() * matrix2.getRows()]);
        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        var rowsPerThread = (int)Math.ceil((float)matrix1.getRows() / maxThreads);

        var threadItems = initThreads(maxThreads);

        var threadNo = 0;

        var actualThreadItems = new ArrayList<ThreadItem>();

        for (var i = 0; i < matrix1.getRows(); i += rowsPerThread) {
            final var rangeStart = i;
            final var rangeLimit = Math.min(i + rowsPerThread, matrix1.getRows());
            var threadItem = threadItems.get(threadNo++);
            threadItem.setFn(() -> multipleTransposedF32Range(resultData, matrix1, matrix2, data1, data2, rangeStart, rangeLimit));
            actualThreadItems.add(threadItem);
        }

        for (var t : actualThreadItems) {
            t.waitTask();
        }

        return result;
    }

    public static void normalize(MatrixF32Interface vector) {
        final var data = vector.getData();
        var max = 0.0f;
        var min = 0.0f;

        for (var i = 0; i < data.length; i++) {
            max = Math.max(max, data[i]);
            min = Math.min(min, data[i]);
        }

        for (var i = 0; i < data.length; i++) {
            data[i] = (data[i] - min) / (max - min);
        }
    }

    public static void softmax(MatrixF32Interface vector, float alpha) {
        final var data = vector.getData();
        var sum = 0.0f;

        for (var i = 0; i < data.length; i++) {
            var exp = normalize((float)Math.exp(data[i] * alpha), Float.MAX_VALUE / vector.getData().length);
            sum += exp;
            data[i] = exp;
        }

        sum = normalize(sum, Float.MAX_VALUE);

        if (sum == 0.0f) {
            Arrays.fill(data, 0.0f);

            return;
        }

        for (var i = 0; i < data.length; i++) {
            data[i] /= sum;
        }
    }

    private static float normalize(float x, float max) {
        return Math.max(Math.min(x, max), -max);
    }

    public static void softmaxDiff(MatrixF32Interface vector, float alpha) {
        final var data = vector.getData();
        var sum = 0.0f;

        for (var i = 0; i < data.length; i++) {
            var exp = (float)Math.max(Math.min(Math.exp(data[i] * alpha), Float.MAX_VALUE / vector.getData().length / 2), -Float.MAX_VALUE / vector.getData().length / 2);
            sum += exp;
            data[i] = exp;
        }

        sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

        if (sum == 0.0f) {
            Arrays.fill(data, 0.0f);

            return;
        }

        for (var i = 0; i < data.length; i++) {
            data[i] = (data[i] / sum) * (1 - data[i] / sum);
        }
    }

    public static void reLU(MatrixF32Interface vector) {
        final var data = vector.getData();

        for (var i = 0; i < data.length; i++) {
            //noinspection ManualMinMaxCalculation
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
    }

    public static void generalizationApply(float l1, MatrixF32Interface weights, float generalizationFactor) {
        var a = weights.getData();
        for (var i = 0; i < a.length; i++) {
            a[i] = a[i] > 0 ? Math.max(0, a[i] - l1 * generalizationFactor) : Math.min(0, a[i] + l1 * generalizationFactor);
        }
    }

    public static float loss(float[] result, float[] target, float threshold) {
        var a = getAnswer(target);

        var loss = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (a == i) {
                continue;
            }

            loss = Math.abs(Math.max(0, result[i] - result[a] + threshold));
        }

        return loss / result.length;
    }

    public static void dropout(Random random, float[] result, float k) {
        for (var i : random.ints((long)(-result.length * Math.log(1 - k)), 0, result.length).toArray()) {
            result[i] = 0.0f;
        }
    }

    public static float dropoutRate(float k) {
        return (1.0f / (1 - k));
    }

    public static int getAnswer(float[] result) {
        var a = 0;
        var max = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (result[i] > max) {
                max = result[i];
                a = i;
            }
        }

        return a;
    }
    public static float generalizeLasso(MatrixF32Interface weights) {
        var result = 0.0f;

        for (float item : weights.getData()) {
            result += Math.abs(item);
        }

        return result / weights.getData().length;
    }

    public static float generalizeRidge(MatrixF32Interface weights) {
        var result = 0.0f;

        for (float item : weights.getData()) {
            result += item * item;
        }

        return result / weights.getData().length;
    }

    private static void multipleF32RangeBlas(float[] resultData, MatrixF32Interface matrix1, MatrixF32Interface matrix2, float[] data1, float[] data2, float alpha, float beta) {
        Blas.getInstance(true).sgemm("N", "N",
                matrix2.getColumns(), matrix1.getRows(),
                matrix2.getRows(), alpha, data2, 0, matrix2.getColumns(),
                data1, 0, matrix1.getColumns(),
                beta,
                resultData, 0, matrix2.getColumns());
    }

    private static void multipleF32Range(float[] resultData, MatrixF32Interface matrix1, MatrixF32Interface matrix2, float[] data1, float[] data2, int rangeStart, int rangeLimit) {
        for (var i = rangeStart; i < rangeLimit; i++) {
            int i1 = i * matrix2.getColumns();
            int i2 = i * matrix1.getColumns();
            for (var j = 0; j < matrix2.getColumns(); j++) {
                for (var k = 0; k < matrix1.getColumns(); k++) {
                    resultData[i1 + j] += data1[i2 + k] * data2[k * matrix2.getColumns() + j];
                }
            }
        }
    }

    private static void multipleTransposedF32Range(float[] resultData, MatrixF32Interface matrix1, MatrixF32Interface matrix2, float[] data1, float[] data2, int rangeStart, int rangeLimit) {
        for (var i = rangeStart; i < rangeLimit; i++) {
            int i1 = i * matrix2.getRows();
            int i2 = i * matrix1.getColumns();
            for (var j = 0; j < matrix2.getRows(); j++) {
                for (var k = 0; k < matrix1.getColumns(); k++) {
                    resultData[i1 + j] += data1[i2 + k] * data2[j * matrix2.getColumns() + k];
                }
            }
        }
    }
}
