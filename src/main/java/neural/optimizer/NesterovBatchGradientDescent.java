package neural.optimizer;

import linear.MatrixF32;
import linear.Ops;
import neural.Layer;

/**
 * Имеет несколько лучшую сходимость на большом количестве слоев
 */
public class NesterovBatchGradientDescent extends BatchGradientDescent {
    private record MomentumItem(MatrixF32 m, MatrixF32 g) {}
    private MomentumItem[] momentumData;
    private float alpha = 0.7f;

    public NesterovBatchGradientDescent() {}

    public NesterovBatchGradientDescent(float alpha) {
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

        float[] gData = sgdItem.gradient.getData();

        MatrixF32 gradientMatrix = new MatrixF32(batchSize, layer.size, gData).transpose();
        float alpha1 = -eta * sgdItem.loss * layer.dropout.getRate() * (1 - alpha);

        Ops.multiple(gradientMatrix, inputResult, momentum.g, alpha1, 0.0f);

//        var species = Ops.species;
//        var upperBound = species.loopBound(momentum.g.getData().length);
//        int length = species.length();
//        for (var j = 0; j < upperBound; j += length) {
//            var g = FloatVector.fromArray(species, momentum.g.getData(), j);
//            var m = FloatVector.fromArray(species, momentum.m.getData(), j);
//            var w = FloatVector.fromArray(species, layer.weights.getData(), j);
//
//            var m2 = g.add(m.mul(alpha));
//            var w2 = g.add(m2.mul(alpha)).add(w);
//
//            m2.intoArray(momentum.m.getData(), j);
//            w2.intoArray(layer.weights.getData(), j);
//        }

//        Ops.multiple(gradientMatrix, inputResult, momentum.m, alpha1, alpha);

        Ops.multiple(momentum.m, alpha);
        Ops.add(momentum.g, momentum.m, 1.0f);
        Ops.add(momentum.g, layer.weights, 1.0f);
        Ops.add(momentum.m, layer.weights, alpha);
    }

    protected void initStaticMemory(Layer[] layers, int batchSize) {
        momentumData = new MomentumItem[layers.length];
        super.initStaticMemory(layers, batchSize);
    }

    protected void initLayerStaticMemory(int i, Layer layer, int batchSize) {
        var w = layer.weights;
        if (w != null) {
            momentumData[i] = new MomentumItem(new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]), new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]));
        }
        super.initLayerStaticMemory(i, layer, batchSize);
    }
}
