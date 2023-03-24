package neural.optimizer;

import jdk.incubator.vector.FloatVector;
import linear.MatrixF32;
import linear.Ops;
import neural.Layer;

/**
 * Примитивный адаптивный алгоритм: уменьшает скорость обновления часто обновляемых весов
 */
public class AdaGradBatch extends BatchGradientDescent {
    private record MomentumItem(MatrixF32 m, MatrixF32 g) {}
    private MomentumItem[] momentumData;

    public AdaGradBatch() {}

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
        float alpha1 = sgdItem.loss * layer.dropout.getRate();

        Ops.multiple(gradientMatrix, inputResult, momentum.g, alpha1, 0.0f);
        Ops.multipleBand(momentum.g.asVector(), momentum.g.asVector(), momentum.m.asVector(), 1.0f, 1.0f);

        var species = Ops.species;
        var upperBound = species.loopBound(momentum.m.getSize());
        int length = species.length();
        var momentumData = momentum.m.getData();
        var gradientData = momentum.g.getData();
        var outputData = layer.weights.getData();

        for (var j = 0; j < upperBound; j += length) {
            var m = FloatVector.fromArray(species, momentumData, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            var o2 = g.div(m.add(1e-7f).sqrt()).mul(-eta).add(w);

            o2.intoArray(outputData, j);
        }
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
