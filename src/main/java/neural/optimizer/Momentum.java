package neural.optimizer;

import linear.Ops;
import linear.VectorF32;
import neural.Optimizer;

import java.util.HashMap;

public class Momentum implements Optimizer.Interface {
    private final HashMap<Integer, VectorF32> momentumData = new HashMap<>();
    private float alpha = 0.9f;

    public Momentum() {}

    public Momentum(float alpha) {
        this.alpha = alpha;
    }

    public void apply(int layer, VectorF32 weights, VectorF32 gradient, float eta) {
        if (momentumData.get(layer) == null || momentumData.get(layer).getSize() != gradient.getSize()) {
            momentumData.put(layer, new VectorF32(gradient.getSize()));
        }

        var m = momentumData.get(layer);

        Ops.multiple(gradient.toVerticalMatrix(), new VectorF32(new float[]{1}).toHorizontalMatrix(), m.toVerticalMatrix(), 1 - alpha, alpha);

        Ops.add(m, weights, -eta);
    }
}
