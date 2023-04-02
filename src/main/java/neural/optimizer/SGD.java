package neural.optimizer;

import linear.Ops;
import linear.VectorF32;
import neural.Optimizer;

public class SGD implements Optimizer.Interface {
    public void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta) {
        Ops.add(gradient, weights, -eta);
    }
}
