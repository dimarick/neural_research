package neural;

import linear.MatrixF32;
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
