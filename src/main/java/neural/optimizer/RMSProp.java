package neural.optimizer;

import linear.Ops;
import linear.VectorF32;
import neural.Optimizer;

public class RMSProp implements Optimizer.Interface2 {
    public void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta) {
        Ops.add(weights, gradient, -eta);
    }
}