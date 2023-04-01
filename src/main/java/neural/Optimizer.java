package neural;

import linear.MatrixF32;
import linear.VectorF32;

public class Optimizer {
    public interface Interface {
        float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta);
    }
    public interface Interface2 {
        void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta);
    }
    public interface BatchInterface {
        float apply(Layer[] layers, MatrixF32[] layerResults, MatrixF32 target, float eta);
    }

    public final static class StochasticGradientDescent extends neural.optimizer.StochasticGradientDescent {}
}
