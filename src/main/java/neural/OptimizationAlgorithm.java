package neural;

import linear.VectorF32;

public interface OptimizationAlgorithm {
    void apply(VectorF32 result);
}
