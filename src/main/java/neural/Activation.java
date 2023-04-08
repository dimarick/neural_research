package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.util.Arrays;

public class Activation {

    public interface Interface {
        VectorF32 apply(VectorF32 vector);
        VectorF32 diff(VectorF32 vector, VectorF32 output);
        MatrixF32 applyBatch(MatrixF32 matrix);
        MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output);
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
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            final var data = matrix.getData();

            for (var i = 0; i < data.length; i++) {
                //noinspection ManualMinMaxCalculation
                data[i] = data[i] > 0.0f ? data[i] : 0.0f;
            }

            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            final var data = matrix.getData();

            for (var i = 0; i < data.length; i++) {
                data[i] = data[i] > 0.0f ? 1.0f : 0.0f;
            }

            return matrix;
        }

        @Override
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            final var data = output.getData();
            final var input = vector.getData();

            for (var i = 0; i < data.length; i++) {
                data[i] = input[i] > 0.0f ? 1.0f : 0.0f;
            }

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.HuberLoss();
        }
    }

    public static class LeakyReLU implements Interface {

        @Override
        public VectorF32 apply(VectorF32 vector) {
            final var data = vector.getData();

            for (var i = 0; i < data.length; i++) {
                data[i] = data[i] > 0.0f ? data[i] : data[i] * 0.01f;
            }

            return vector;
        }

        @Override
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            final var data = output.getData();
            final var input = vector.getData();

            for (var i = 0; i < data.length; i++) {
                data[i] = input[i] > 0.0f ? 1.0f : 0.01f;
            }

            return output;
        }

        @Override
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            final var data = matrix.getData();

            for (var i = 0; i < data.length; i++) {
                data[i] = data[i] > 0.0f ? data[i] : 0.01f * data[i];
            }

            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            final var data = matrix.getData();
            final var o = output.getData();

            for (var i = 0; i < data.length; i++) {
                o[i] = data[i] > 0.0f ? 1.0f : 0.01f;
            }

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.HuberLoss();
        }
    }

    public static class SReLU implements Interface {

        private float alpha1 = 0.1f;
        private float alpha2 = 1f;
        private float alpha3 = 0.1f;

        private float x1 = -1f;
        private float x2 = 1f;

        public SReLU() {}

        public SReLU(float alpha1, float alpha2, float alpha3, float x1, float x2) {
            this.alpha1 = alpha1;
            this.alpha2 = alpha2;
            this.alpha3 = alpha3;
            this.x1 = x1;
            this.x2 = x2;
        }

        @Override
        public VectorF32 apply(VectorF32 vector) {
            throw new RuntimeException();
        }

        @Override
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            throw new RuntimeException();
        }

        @Override
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            final var data = matrix.getData();

            for (var i = 0; i < data.length; i++) {
                float x = data[i];

                if (x < x1) {
                    data[i] = (x - x1) * alpha1 + x1 * alpha2;
                } else if (x < x2) {
                    data[i] = x * alpha2;
                } else {
                    data[i] = (x - x2) * alpha3 + x2 * alpha2;
                }
            }

            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            final var data = output.getData();
            final var input = matrix.getData();

            for (var i = 0; i < data.length; i++) {
                float x = input[i];
                if (x < x1) {
                    data[i] = alpha1;
                } else if (x < x2) {
                    data[i] = alpha2;
                } else {
                    data[i] = alpha3;
                }
            }

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.HuberLoss();
        }
    }

    public static class Softmax implements Interface {

        final private float alpha;
        private final float e = 1e-10f;

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
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            final var data = output.getData();
            final var input = vector.getData();
            var sum = 0.0f;

            for (var i = 0; i < data.length; i++) {
                var exp = (float)Math.max(Math.min(Math.exp(input[i] * alpha), Float.MAX_VALUE / input.length / 2), -Float.MAX_VALUE / input.length / 2);
                sum += exp;
                data[i] = exp;
            }

            sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

            if (sum == 0.0f) {
                Arrays.fill(data, 0.0f);

                return new VectorF32(data);
            }

            for (var i = 0; i < data.length; i++) {
                data[i] = (input[i] / sum) * (1 - input[i] / sum);
            }

            return output;
        }

        @Override
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            final var data = matrix.getData();

            for (var i = 0; i < matrix.getRows(); i++) {
                var sum = 0.0f;
                var k = i * matrix.getColumns();

                for (var j = 0; j < matrix.getColumns(); j++) {
                    var exp = normalize((float) Math.exp(data[k + j] * alpha), Float.MAX_VALUE / matrix.getColumns());
                    sum += exp;
                    data[k + j] = exp;
                }

                sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

                for (var j = 0; j < matrix.getColumns(); j++) {
                    data[k + j] /= sum + e;
                    Ops.assertNoNan(new float[]{data[k + j]});
                }
            }

            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            final var data = output.getData();
            final var input = matrix.getData();

            for (var i = 0; i < matrix.getRows(); i++) {
                var sum = 0.0f;
                var k = i * matrix.getColumns();

                for (var j = 0; j < matrix.getColumns(); j++) {
                    var exp = (float) Math.max(Math.min(Math.exp(input[k + j] * alpha), Float.MAX_VALUE / input.length / 2), -Float.MAX_VALUE / input.length / 2);
                    sum += exp;
                    data[k + j] = exp;
                }

                sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

                if (sum == 0.0f) {
                    Arrays.fill(data, 0.0f);
                }

                for (var j = 0; j < data.length; j++) {
                    data[j] = (input[j] / sum) * (1 - input[j] / sum);
                }
            }

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.CrossEntropyLoss();
        }

        private static float normalize(float x, float max) {
            return Math.max(Math.min(x, max), -max);
        }
    }

    public static class SoftmaxStable2 implements Interface {

        private float alpha = 1f;
        private float C = 1f;

        public SoftmaxStable2(float alpha, float C) {
            this.alpha = alpha;
            this.C = C;
        }

        public SoftmaxStable2() {

        }

        @Override
        public VectorF32 apply(VectorF32 vector) {
            throw new RuntimeException();
        }

        @Override
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            throw new RuntimeException();
        }

        @Override
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            final var data = matrix.getData();

            for (var i = 0; i < matrix.getRows(); i++) {
                var sum = (float)Math.exp(C);
                var k = i * matrix.getColumns();

                for (var j = 0; j < matrix.getColumns(); j++) {
                    var exp = (float)Math.exp(data[k + j] * alpha);
                    sum += exp;
                    data[k + j] = exp;
                }

                for (var j = 0; j < matrix.getColumns(); j++) {
                    data[k + j] *= 2f / sum;
                    Ops.assertNoNan(new float[]{data[k + j]});
                }
            }

            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            final var data = output.getData();
            final var input = matrix.getData();

            for (var i = 0; i < matrix.getRows(); i++) {
                var sum = (float)Math.exp(C);
                var k = i * matrix.getColumns();

                for (var j = 0; j < matrix.getColumns(); j++) {
                    var exp = (float) Math.max(Math.min(Math.exp(input[k + j] * alpha), Float.MAX_VALUE / input.length / 2), -Float.MAX_VALUE / input.length / 2);
                    sum += exp;
                    data[k + j] = exp;
                }

                sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

                if (sum == 0.0f) {
                    Arrays.fill(data, 0.0f);
                }

                for (var j = 0; j < data.length; j++) {
                    data[j] = (input[j] / sum) * (1 - input[j] / sum);
                }
            }

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.CrossEntropyLoss();
        }
    }

    public static class SoftmaxStable implements Interface {

        final private float alpha;

        public SoftmaxStable(float alpha) {
            this.alpha = alpha;
        }

        public SoftmaxStable() {
            this.alpha = 1.0f;
        }

        @Override
        public VectorF32 apply(VectorF32 vector) {
            throw new RuntimeException();
        }

        @Override
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            throw new RuntimeException();
        }

        @Override
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            final var data = matrix.getData();

            for (var i = 0; i < matrix.getRows(); i++) {
                var sum = 0.0f;
                var k = i * matrix.getColumns();

                var max = Ops.max(data, k, matrix.getColumns());

                for (var j = 0; j < matrix.getColumns(); j++) {
                    var exp = (float)Math.exp(data[k + j] - max);
                    sum += exp;
                    data[k + j] = exp;
                }

                for (var j = 0; j < matrix.getColumns(); j++) {
                    data[k + j] /= sum;
                }
            }

            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            final var data = output.getData();
            final var input = matrix.getData();

            for (var i = 0; i < matrix.getRows(); i++) {
                var sum = 0.0f;
                var k = i * matrix.getColumns();

                for (var j = 0; j < matrix.getColumns(); j++) {
                    var exp = (float) Math.max(Math.min(Math.exp(input[k + j] * alpha), Float.MAX_VALUE / input.length / 2), -Float.MAX_VALUE / input.length / 2);
                    sum += exp;
                    data[k + j] = exp;
                }

                sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

                if (sum == 0.0f) {
                    Arrays.fill(data, 0.0f);
                }

                for (var j = 0; j < data.length; j++) {
                    data[j] = (input[j] / sum) * (1 - input[j] / sum);
                }
            }

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.CrossEntropyLoss();
        }
    }


    public static class Linear implements Interface {

        @Override
        public VectorF32 apply(VectorF32 vector) {
            return vector;
        }

        @Override
        public VectorF32 diff(VectorF32 vector, VectorF32 output) {
            Arrays.fill(output.getData(), 1f);

            return output;
        }

        @Override
        public MatrixF32 applyBatch(MatrixF32 matrix) {
            return matrix;
        }

        @Override
        public MatrixF32 diffBatch(MatrixF32 matrix, MatrixF32 output) {
            Arrays.fill(output.getData(), 1f);

            return output;
        }

        @Override
        public Loss.Interface suggestLoss() {
            return new Loss.HuberLoss();
        }
    }
}
