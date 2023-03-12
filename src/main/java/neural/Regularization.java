package neural;

import dev.ludovic.netlib.BLAS;
import linear.MatrixF32;

public class Regularization {
    public interface Interface {
        void apply(MatrixF32 weights);
    }

    public static class Lasso implements Interface {
        private float generalizationFactor = 1e-5f;

        public Lasso() {}

        public Lasso(float generalizationFactor) {
            this.generalizationFactor = generalizationFactor;
        }

        public void apply(MatrixF32 weights) {
            float[] data = weights.getData();

            var l = BLAS.getInstance().sasum(data.length, data, 1) / data.length;
            generalizationApply(l, weights, generalizationFactor);
        }
    }

    public static class Ridge implements Interface {
        private float generalizationFactor = 1e-5f;

        public Ridge() {}

        public Ridge(float generalizationFactor) {
            this.generalizationFactor = generalizationFactor;
        }

        public void apply(MatrixF32 weights) {
            var result = 0.0f;

            for (float item : weights.getData()) {
                result += item * item;
            }

            var l = result / weights.getData().length;
            generalizationApply(l, weights, generalizationFactor);
        }
    }

    private static void generalizationApply(float l1, MatrixF32 weights, float generalizationFactor) {
        var a = weights.getData();
        for (var i = 0; i < a.length; i++) {
            a[i] = a[i] > 0 ? Math.max(0, a[i] - l1 * generalizationFactor) : Math.min(0, a[i] + l1 * generalizationFactor);
        }
    }
}
