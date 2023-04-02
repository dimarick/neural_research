package neural;

import linear.VectorF32;

public class Optimizer {
    public interface Interface {
        void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta);
    }
}
