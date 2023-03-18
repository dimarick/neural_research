package neural;

import linear.VectorF32;

public class Loss {
    public interface Interface {
        float apply(VectorF32 error, VectorF32 predicted);
    }

    public static class CrossEntropyLoss implements Interface {
        public float apply(VectorF32 error, VectorF32 predicted) {
            var loss = 0.0f;

            float[] errorData = error.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < errorData.length; i++) {
                loss += errorData[i] * Math.log(1e-15 + predictedData[i]);
            }

            return -loss / error.getSize();
        }
    }

    public static class MeanAbsoluteErrorLoss implements Interface {
        public float apply(VectorF32 error, VectorF32 predicted) {
            var loss = 0.0f;

            float[] errorData = error.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < errorData.length; i++) {
                loss += Math.abs(errorData[i] - predictedData[i]);
            }

            return loss / error.getSize();
        }
    }

    public static class MeanSquaredErrorLoss implements Interface {
        public float apply(VectorF32 error, VectorF32 predicted) {
            var loss = 0.0f;

            float[] errorData = error.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < errorData.length; i++) {
                float e = errorData[i] - predictedData[i];
                loss += e * e;
            }

            return loss / error.getSize();
        }
    }

    public static class HuberLoss implements Interface {
        private float delta = 1.0f;

        public HuberLoss() {}
        public HuberLoss(float delta) {
            this.delta = delta;
        }

        public float apply(VectorF32 error, VectorF32 predicted) {
            var loss = 0.0f;

            float[] errorData = error.getData();

            for (var i = 0; i < errorData.length; i++) {
                float e = Math.abs(errorData[i]);
                if (e > delta) {
                    loss += e * delta - 0.5 * delta * delta;
                } else {
                    loss += 0.5 * e * e;
                }
            }

            return loss / error.getSize();
        }
    }
}
