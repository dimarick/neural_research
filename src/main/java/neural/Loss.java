package neural;

import linear.VectorF32;

public class Loss {
    public interface Interface {
        float apply(VectorF32 target, VectorF32 predicted);
    }

    public static class CrossEntropyLoss implements Interface {
        public float apply(VectorF32 target, VectorF32 predicted) {
            var loss = 0.0f;

            float[] targetData = target.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < targetData.length; i++) {
                loss += targetData[i] * Math.log(1e-15 + predictedData[i]);
            }

            return -loss / target.getSize();
        }
    }

    public static class MeanAbsoluteErrorLoss implements Interface {
        public float apply(VectorF32 target, VectorF32 predicted) {
            var loss = 0.0f;

            float[] targetData = target.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < targetData.length; i++) {
                loss += Math.abs(targetData[i] - predictedData[i]);
            }

            return loss / target.getSize();
        }
    }

    public static class MeanSquaredErrorLoss implements Interface {
        public float apply(VectorF32 target, VectorF32 predicted) {
            var loss = 0.0f;

            float[] targetData = target.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < targetData.length; i++) {
                float e = targetData[i] - predictedData[i];
                loss += e * e;
            }

            return loss / target.getSize();
        }
    }

    public static class HuberLoss implements Interface {
        private float delta = 1.0f;

        public HuberLoss() {}
        public HuberLoss(float delta) {
            this.delta = delta;
        }

        public float apply(VectorF32 target, VectorF32 predicted) {
            var loss = 0.0f;

            float[] targetData = target.getData();
            float[] predictedData = predicted.getData();

            for (var i = 0; i < targetData.length; i++) {
                float e = Math.abs(targetData[i] - predictedData[i]);
                if (e > delta) {
                    loss += e * delta - 0.5 * delta * delta;
                } else {
                    loss += 0.5 * e * e;
                }
            }

            return loss / target.getSize();
        }
    }
}
