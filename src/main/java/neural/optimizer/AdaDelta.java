package neural.optimizer;

import jdk.incubator.vector.FloatVector;
import linear.Ops;
import linear.VectorF32;
import neural.Optimizer;

import java.util.Arrays;
import java.util.HashMap;

public class AdaDelta implements Optimizer.Interface {
    private final HashMap<Integer, VectorF32> gData = new HashMap<>();
    private final HashMap<Integer, VectorF32> dData = new HashMap<>();
    private float alpha = 0.9f;

    public AdaDelta() {}

    public AdaDelta(float alpha) {
        this.alpha = alpha;
    }
    public void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta) {
        if (gData.get(layer) == null || gData.get(layer).getSize() != gradient.getSize()) {
            gData.put(layer, new VectorF32(gradient.getSize()));
            VectorF32 dVector = new VectorF32(gradient.getSize());
            Arrays.fill(dVector.getData(), 1f);
            dData.put(layer, dVector);
        }

        var gDataItem = gData.get(layer).getData();
        var deltaData = dData.get(layer).getData();

        var species = Ops.species;
        var upperBound = species.loopBound(gDataItem.length);

        int length = species.length();
        var gradientData = gradient.getData();
        var outputData = weights.getData();

        for (var j = 0; j < upperBound; j += length) {
            var G = FloatVector.fromArray(species, gDataItem, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var d = FloatVector.fromArray(species, deltaData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            G = g.mul(g).mul(1 - alpha).add(G.mul(alpha));

            var o2 = g
                    .mul(d.sqrt().add(1e-10f))
                    .div(G.sqrt().add(1e-10f));

            o2.mul(-eta).add(w).intoArray(outputData, j);
            o2.mul(o2).mul(1 - alpha).add(d.mul(alpha)).intoArray(deltaData, j);
            G.intoArray(gDataItem, j);
        }
    }
}
