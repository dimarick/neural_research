package neural;

import linear.VectorF32;

public class Optimizer {
    public interface Interface {
        float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta);
    }

    public final static class StochasticGradientDescent extends neural.optimizer.StochasticGradientDescent {}
}
