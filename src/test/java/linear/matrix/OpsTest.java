package linear.matrix;

import nu.pattern.OpenCV;
import org.jblas.FloatMatrix;
import org.jblas.NativeBlas;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import java.util.Arrays;

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
    }
    @Test
    public void multipleOpenCVTest() {
        OpenCV.loadLocally();

        var mat1 = new Mat(3, 2, CvType.CV_32F);
        mat1.put(new int[]{0,0}, new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f});

        var mat2 = new Mat(2, 4, CvType.CV_32F);
        mat2.put(new int[]{0,0}, new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});

        var result = new float[12];
        mat1.matMul(mat2).get(new int[]{0,0}, result);

        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                result,
                0.0f
        );
    }
    @Test
    public void multipleOpenCVLargeTest() {
        OpenCV.loadLocally();

        var mat1 = new Mat(4000, 1000, CvType.CV_32F);
        mat1.put(new int[]{0,0}, new float[4000000]);

        var mat2 = new Mat(1000, 4000, CvType.CV_32F);
        mat2.put(new int[]{0,0}, new float[4000000]);

        mat1.matMul(mat2);
        mat1.matMul(mat2);
    }
    @Test
    public void multipleOpenCVLargeFloat64Test() {
        OpenCV.loadLocally();

        var mat1 = new Mat(4000, 1000, CvType.CV_64F);
        mat1.put(new int[]{0,0}, new double[4000000]);

        var mat2 = new Mat(1000, 4000, CvType.CV_64F);
        mat2.put(new int[]{0,0}, new double[4000000]);

        mat1.matMul(mat2);
        mat1.matMul(mat2);
    }
    @Test
    public void multipleMKLTest() {
        FloatMatrix c = new FloatMatrix(4, 3);

        NativeBlas.sgemm('N', 'N', 4, 3, 2, 1.0f, new float[]{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 1,
                4, new float[]{0.0f, 0.0f, 0.0f, 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, 3, 2, 0.0f, c.data, 0, 4);

        var c1 = new FloatMatrix(4, 3);

        SimpleBlas.gemm(
                1.0f,
                new FloatMatrix(4, 2, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f),
                new FloatMatrix(2, 3, 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f),
                0.0f,
                c1
        );

        SimpleBlas.gemm(
                1.0f,
                new FloatMatrix(4, 2, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f),
                new FloatMatrix(2, 3, 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f),
                0.0f,
                c1
        );
        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                c1.data,
                0.0f
        );
        assertArrayEquals(
                new float[]{21.0f,26.0f,31.0f,36.0f,27.0f,34.0f,41.0f,48.0f,33.0f,42.0f,51.0f,60.0f},
                c.data,
                0.0f
        );
    }
    @Test
    public void multipleMKLLargeFloatTest() {
        FloatMatrix c = new FloatMatrix(4000, 4000);
        SimpleBlas.gemm(1.0f, FloatMatrix.ones(4000, 10000), FloatMatrix.ones(10000, 4000), 0.0f, c);
        SimpleBlas.gemm(1.0f, FloatMatrix.ones(4000, 10000), FloatMatrix.ones(10000, 4000), 0.0f, c);
    }
    @Test
    public void multipleLargeTest() {
        float[] data1 = new float[400000000];
        float[] data2 = new float[400000000];
        Arrays.fill(data1, 1.0f);
        Arrays.fill(data2, 1.0f);

        float[] result = Ops.multiple(
                new MatrixF32(4000, 100000, data1),
                new MatrixF32(100000, 4000, data2)
        ).getData();
        assertEquals(
                16000000,
                result.length
        );

        assertEquals(100000f, result[0], 0.0f);
        assertEquals(100000f, result[16000000 - 1], 0.0f);
    }

    @Test
    public void multipleLargeVectorTest() {
        float[] data1 = new float[400000000];
        float[] data2 = new float[10000];
        Arrays.fill(data1, 1.0f);
        Arrays.fill(data2, 1.0f);

        float[] result = new float[0];

        for (var i = 0; i < 20; i++) {
            result = Ops.multiple(
                    new MatrixF32(40000, 10000, data1),
                    new MatrixF32(10000, 1, data2)
            ).getData();
        }

        assertEquals(
                40000,
                result.length
        );

        assertEquals(10000f, result[0], 0.0f);
        assertEquals(10000f, result[40000 - 1], 0.0f);
    }

    @Test
    public void multipleMultiLargeTest() {
        MatrixF32 matrix1 = new MatrixF32(500, 1000, new float[500000]);
        MatrixF32 matrix2 = new MatrixF32(1000, 2000, new float[2000000]);

        for (var i = 0; i < 10; i++) {
            Ops.multiple(matrix1, matrix2);
        }
    }
}
