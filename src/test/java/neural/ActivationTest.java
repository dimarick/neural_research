package neural;

import linear.MatrixF32;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class ActivationTest {

    @Test
    public void testSReLU() {
        var relu = new Activation.SReLU(0.1f, 2f, 0.2f, -1f, 2f);
        var result = relu.applyBatch(new MatrixF32(1, 3, new float[]{-3, 1, 3}));
        assertArrayEquals(new float[]{-2.2f, 2f, 4.2f}, result.getData(), 0.1f);
    }
}