package neural.optimizer;

import jdk.incubator.vector.FloatVector;
import linear.Ops;
import linear.VectorF32;
import neural.Optimizer;

import java.util.ArrayList;

public class AdaGrad implements Optimizer.Interface2 {
    private final ArrayList<VectorF32> gData = new ArrayList<>();

    public void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta) {
        if (gData.get(layer) == null || gData.get(layer).getSize() != gradient.getSize()) {
            gData.set(layer, new VectorF32(gradient.getSize()));
        }

        var gDataItem = gData.get(layer).getData();

        var species = Ops.species;
        var upperBound = species.loopBound(gDataItem.length);

        int length = species.length();
        var gradientData = gradient.getData();
        var outputData = weights.getData();


        for (var j = 0; j < upperBound; j += length) {
            var G = FloatVector.fromArray(species, gDataItem, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            var o = g.div(G.add(1e-12f).sqrt()).mul(-eta).add(w);
            var o2 = g.mul(g);

            o.intoArray(outputData, j);
            o2.intoArray(gDataItem, j);
        }
    }
}
