package neural;

import linear.MatrixF32;

public interface RegularizingWeightsInterface {
    void apply(MatrixF32 weights);
}
