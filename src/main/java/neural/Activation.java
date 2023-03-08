package neural;

import linear.VectorF32;

public class Activation {

    public interface Interface {
        VectorF32 apply(VectorF32 vector);
        VectorF32 diff(VectorF32 vector);
    }

    public static class ReLU implements Interface {

        @Override
        public VectorF32 apply(VectorF32 vector) {
            NeuralAlgo.reLU(vector);

            return vector;
        }

        @Override
        public VectorF32 diff(VectorF32 vector) {
            return new VectorF32(NeuralAlgo.reLUDiff(vector));
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
            NeuralAlgo.softmax(vector, alpha);

            return vector;
        }

        @Override
        public VectorF32 diff(VectorF32 vector) {
            return new VectorF32(NeuralAlgo.softmaxDiff(vector, alpha));
        }
    }
}
