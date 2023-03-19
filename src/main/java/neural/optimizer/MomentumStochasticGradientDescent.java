package neural.optimizer;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;
import neural.Layer;

public class MomentumStochasticGradientDescent extends StochasticGradientDescent {
    private record MomentumItem(MatrixF32 w) {}
    private MomentumItem[] momentumData;
    private float alpha = 0.9f;

    public MomentumStochasticGradientDescent() {}

    public MomentumStochasticGradientDescent(float alpha) {
        this.alpha = alpha;
    }

    protected void updateLayerWeights(Layer[] layers, VectorF32[] layerResults, float eta, SgdDataItem sgdItem) {
        var i = sgdItem.i;
        var layer = layers[i];
        var momentum = momentumData[i];

        if (momentum == null) {
            super.updateLayerWeights(layers, layerResults, eta, sgdItem);

            return;
        }

        Ops.multiple(sgdItem.gradient, layerResults[i - 1], momentum.w, -eta * sgdItem.loss * layer.dropout.getRate() * (1 - alpha), alpha);
        Ops.add(momentum.w.getData(), layer.weights.getData(), 1.0f);
    }

    protected void initStaticMemory(Layer[] layers) {
        momentumData = new MomentumItem[layers.length];
        super.initStaticMemory(layers);
    }

    protected void initLayerStaticMemory(int i, Layer layer) {
        var w = layer.weights;
        if (w != null) {
            momentumData[i] = new MomentumItem(new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]));
        }
        super.initLayerStaticMemory(i, layer);
    }
}