package neural;

import dev.ludovic.netlib.BLAS;
import linear.MatrixF32;
import linear.Ops;

public class Regularization {
    public interface Interface {
        void apply(MatrixF32 weights, float eta);
    }

    public static class Lasso implements Interface {
        private float generalizationFactor = 1e-5f;

        public Lasso() {}

        public Lasso(float generalizationFactor) {
            this.generalizationFactor = generalizationFactor;
        }

        public void apply(MatrixF32 weights, float eta) {
            float[] data = weights.getData();

            var l = BLAS.getInstance().sasum(data.length, data, 1) / data.length * generalizationFactor;
            generalizationApply(l * eta, weights);
        }
    }

    public static class Ridge implements Interface {
        private float generalizationFactor = 1e-5f;

        public Ridge() {}

        public Ridge(float generalizationFactor) {
            this.generalizationFactor = generalizationFactor;
        }

        public void apply(MatrixF32 weights, float eta) {
            float[] data = weights.getData();

            var l = Ops.multiple(
                    new MatrixF32(1, data.length, data),
                    new MatrixF32(data.length, 1, data), 1.0f / data.length * generalizationFactor, 0
            ).getData()[0];

            generalizationApply(l * eta, weights);
        }
    }

    public static class ElasticNet implements Interface {
        private float generalizationFactor = 1e-5f;

        public ElasticNet() {}

        public ElasticNet(float generalizationFactor) {
            this.generalizationFactor = generalizationFactor;
        }

        public void apply(MatrixF32 weights, float eta) {
            var result = 0.0f;

            for (float item : weights.getData()) {
                float abs = Math.abs(item);
                result += abs < 1 ? 0.5f * item * item : abs;
            }

            var l = result / weights.getData().length * generalizationFactor;
            generalizationApply(l * eta, weights);
        }
    }

    private static void generalizationApply(float l1, MatrixF32 weights) {
        var a = weights.getData();
        for (var i = 0; i < a.length; i++) {
            a[i] = a[i] > 0 ? Math.max(0, a[i] - l1) : (a[i] < 0 ? Math.min(0, a[i] + l1) : 0);
        }
    }
}
