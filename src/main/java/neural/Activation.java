package neural;

import linear.VectorF32;

import java.util.Arrays;

public class Activation {

    public interface Interface {
        VectorF32 apply(VectorF32 vector);
        VectorF32 diff(VectorF32 vector);
        Loss.Interface suggestLoss();
    }

    public static class ReLU implements Interface {

        @Override
        public VectorF32 apply(VectorF32 vector) {
            final var data = vector.getData();

            for (var i = 0; i < data.length; i++) {
                //noinspection ManualMinMaxCalculation
                data[i] = data[i] > 0.0f ? data[i] : 0.0f;
            }

            return vector;
        }
        @Override
        public VectorF32 diff(VectorF32 vector) {
            final var data = vector.getData().clone();

            for (var i = 0; i < data.length; i++) {
                data[i] = data[i] > 0.0f ? 1.0f : 0.0f;
            }

            return new VectorF32(data);
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.HuberLoss();
        }
    }

    public static class Softmax implements Interface {

        final private float alpha;

        public Softmax(float alpha) {
            this.alpha = alpha;
        }

        public Softmax() {
            this.alpha = 1.0f;
        }

        @Override
        public VectorF32 apply(VectorF32 vector) {
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

                return vector;
            }

            for (var i = 0; i < data.length; i++) {
                data[i] /= sum;
            }

            return vector;
        }

        @Override
        public VectorF32 diff(VectorF32 vector) {
            final var data = vector.getData().clone();
            var sum = 0.0f;

            for (var i = 0; i < data.length; i++) {
                var exp = (float)Math.max(Math.min(Math.exp(data[i] * alpha), Float.MAX_VALUE / vector.getData().length / 2), -Float.MAX_VALUE / vector.getData().length / 2);
                sum += exp;
                data[i] = exp;
            }

            sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

            if (sum == 0.0f) {
                Arrays.fill(data, 0.0f);

                return new VectorF32(data);
            }

            for (var i = 0; i < data.length; i++) {
                data[i] = (data[i] / sum) * (1 - data[i] / sum);
            }

            return new VectorF32(data);
        }
        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.CrossEntropyLoss();
        }

        private static float normalize(float x, float max) {
            return Math.max(Math.min(x, max), -max);
        }
    }


    public static class Linear implements Interface {

        @Override
        public VectorF32 apply(VectorF32 vector) {
            return vector;
        }

        @Override
        public VectorF32 diff(VectorF32 vector) {
            var diff = vector.getData().clone();

            Arrays.fill(diff, 1f);

            return new VectorF32(diff);
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.HuberLoss();
        }
    }
}
