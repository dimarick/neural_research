package linear.matrix;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class OpsTest {
    @Test
    public void multipleTest() {
        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                Ops.multiple(
                    new MatrixF32(3, 2, new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}),
                    new MatrixF32(2, 4, new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f})
                ).getData(),
                0.0f
        );
        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                Ops.multipleTransposed(
                        new MatrixF32(3, 2, new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}),
                        new MatrixF32(4, 2, new float[]{1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f})
                ).getData(),
                0.0f
        );
    }
    @Test
    public void multipleConcurrentTest() {
        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                Ops.multipleConcurrent(
                    new MatrixF32(3, 2, new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}),
                    new MatrixF32(2, 4, new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}),
                    3
                ).getData(),
                0.0f
        );
        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                Ops.multipleTransposedConcurrent(
                        new MatrixF32(3, 2, new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}),
                        new MatrixF32(4, 2, new float[]{1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f}),
                        2
                ).getData(),
                0.0f
        );
    }

    @Test
    public void multipleLargeTest() {
        assertEquals(
                16000000,
                Ops.multiple(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(1000, 4000, new float[4000000])
                ).getData().length
        );
        assertEquals(
                16000000,
                Ops.multiple(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(1000, 4000, new float[4000000])
                ).getData().length
        );
    }
    @Test
    public void multipleTransposedLargeTest() {
        assertEquals(
                16000000,
                Ops.multipleTransposed(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(4000, 1000, new float[4000000])
                ).getData().length
        );
        assertEquals(
                16000000,
                Ops.multipleTransposed(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(4000, 1000, new float[4000000])
                ).getData().length
        );
    }

    @Test
    public void multipleLargeConcurrentTest() {
        assertEquals(
                16000000,
                Ops.multipleConcurrent(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(1000, 4000, new float[4000000]),
                        20
                ).getData().length
        );
        assertEquals(
                16000000,
                Ops.multipleConcurrent(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(1000, 4000, new float[4000000]),
                        20
                ).getData().length
        );
    }

    @Test
    public void multipleTransposedLargeConcurrentTest() {
        assertEquals(
                16000000,
                Ops.multipleTransposedConcurrent(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(4000, 1000, new float[4000000]),
                        20
                ).getData().length
        );
        assertEquals(
                16000000,
                Ops.multipleTransposedConcurrent(
                        new MatrixF32(4000, 1000, new float[4000000]),
                        new MatrixF32(4000, 1000, new float[4000000]),
                        20
                ).getData().length
        );
    }

    @Test
    public void multipleMultiLargeTest() {
        MatrixF32 matrix1 = new MatrixF32(500, 1000, new float[500000]);
        MatrixF32 matrix2 = new MatrixF32(1000, 2000, new float[2000000]);

        for (var i = 0; i < 10; i++) {
            Ops.multiple(matrix1, matrix2);
        }
    }

    @Test
    public void multipleTransposedMultiLargeTest() {
        MatrixF32 matrix1 = new MatrixF32(2000, 1000, new float[2000000]);
        MatrixF32 matrix2 = new MatrixF32(500, 1000, new float[500000]);

        for (var i = 0; i < 10; i++) {
            Ops.multipleTransposed(matrix1, matrix2);
        }
    }

    @Test
    public void multipleMultiLargeConcurrentTest() {
        MatrixF32 matrix1 = new MatrixF32(500, 1000, new float[500000]);
        MatrixF32 matrix2 = new MatrixF32(1000, 2000, new float[2000000]);

        for (var i = 0; i < 10; i++) {
            Ops.multipleConcurrent(matrix1, matrix2, 20);
        }
    }

    @Test
    public void multipleTransposedMultiLargeConcurrentTest() {
        MatrixF32 matrix1 = new MatrixF32(2000, 1000, new float[2000000]);
        MatrixF32 matrix2 = new MatrixF32(500, 1000, new float[500000]);

        for (var i = 0; i < 10; i++) {
            Ops.multipleTransposedConcurrent(matrix1, matrix2, 20);
        }
    }
}
