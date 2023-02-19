package linear.matrix;

import java.util.ArrayList;

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
        if (matrix1.getColumns() != matrix2.getRows()) {
            throw new ArrayIndexOutOfBoundsException("incompatible matrix");
        }

        var result = new MatrixF32(matrix1.getRows(), matrix2.getColumns(), new float[matrix1.getRows() * matrix2.getColumns()]);
        float[] data1 = matrix1.getData();
        float[] data2 = matrix2.getData();
        float[] resultData = result.getData();

        multipleF32Range(resultData, matrix1, matrix2, data1, data2, 0, matrix1.getRows());

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

    private static MatrixF32Interface transposeVector(MatrixF32Interface matrix) {
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

    public static void logisticFn(MatrixF32Interface vector) {
        final var data = vector.getData();

        for (var i = 0; i < data.length; i++) {
            data[i] = 1 / (1 + (float)Math.exp(-data[i]));
        }
    }

    public static void reLU(MatrixF32Interface vector) {
        final var data = vector.getData();

        for (var i = 0; i < data.length; i++) {
            //noinspection ManualMinMaxCalculation
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
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
