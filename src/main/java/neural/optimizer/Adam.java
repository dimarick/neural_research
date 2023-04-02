package neural.optimizer;

import jdk.incubator.vector.FloatVector;
import linear.Ops;
import linear.VectorF32;
import neural.Optimizer;

import java.util.HashMap;

public class Adam implements Optimizer.Interface {
    private final HashMap<Integer, VectorF32> mData = new HashMap<>();
    private final HashMap<Integer, VectorF32> vData = new HashMap<>();
    private final HashMap<Integer, Float> layerBeta1 = new HashMap<>();
    private final HashMap<Integer, Float> layerBeta2 = new HashMap<>();
    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    public Adam() {}
    public void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta) {
        if (mData.get(layer) == null || mData.get(layer).getSize() != gradient.getSize()) {
            mData.put(layer, new VectorF32(gradient.getSize()));
            vData.put(layer, new VectorF32(gradient.getSize()));
            layerBeta1.put(layer, beta1);
            layerBeta2.put(layer, beta2);
        }

        var mDataItem = mData.get(layer).getData();

        var species = Ops.species;
        var upperBound = species.loopBound(mDataItem.length);

        int length = species.length();
        var gradientData = gradient.getData();
        var outputData = weights.getData();
        var velocityData = vData.get(layer).getData();

        Float b1 = layerBeta1.get(layer);
        Float b2 = layerBeta2.get(layer);

        for (var j = 0; j < upperBound; j += length) {
            var m = FloatVector.fromArray(species, mDataItem, j);
            var v = FloatVector.fromArray(species, velocityData, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            m = g.mul(1 - beta1).add(m.mul(beta1));
            v = g.mul(g).mul(1 - beta2).add(v.mul(beta2));

            var o =
                    m.div(1 - b1).mul(-eta)
                        .div(v.div(1 - b2).sqrt().add(1e-10f))
                        .add(w);

            o.intoArray(outputData, j);
            m.intoArray(mDataItem, j);
            v.intoArray(velocityData, j);
        }

        layerBeta1.put(layer, b1 * beta1);
        layerBeta2.put(layer, b2 * beta2);
    }
}
