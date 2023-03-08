package neural;

import linear.VectorF32;

public interface LossFunctionInterface {
    VectorF32 apply(VectorF32 vector);
}
