package neural.optimizer;

import linear.MatrixF32;
import linear.Ops;
import neural.Layer;

public class MomentumBatchGradientDescent extends BatchGradientDescent {
    private record MomentumItem(MatrixF32 w) {}
    private MomentumItem[] momentumData;
    private float alpha = 0.9f;

    public MomentumBatchGradientDescent() {}

    public MomentumBatchGradientDescent(float alpha) {
        this.alpha = alpha;
    }

    protected void updateLayerWeights(Layer[] layers, MatrixF32[] layerResults, float eta, SgdDataItem sgdItem) {
        var i = sgdItem.i;
        var layer = layers[i];
        var momentum = momentumData[i];

        if (momentum == null) {
            super.updateLayerWeights(layers, layerResults, eta, sgdItem);

            return;
        }

        MatrixF32 inputResult = layerResults[i - 1];
        var batchSize = inputResult.getRows();
        MatrixF32 gradientMatrix = new MatrixF32(batchSize, layer.size, sgdItem.gradient.getData()).transpose();
        Ops.multiple(gradientMatrix, inputResult, momentum.w, -eta * sgdItem.loss * layer.dropout.getRate() * (1 - alpha), alpha);
        Ops.add(momentum.w.getData(), layer.weights.getData(), 1.0f);
    }

    protected void initStaticMemory(Layer[] layers, int batchSize) {
        momentumData = new MomentumItem[layers.length];
        super.initStaticMemory(layers, batchSize);
    }

    protected void initLayerStaticMemory(int i, Layer layer, int batchSize) {
        var w = layer.weights;
        if (w != null) {
            momentumData[i] = new MomentumItem(new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]));
        }
        super.initLayerStaticMemory(i, layer, batchSize);
    }
}