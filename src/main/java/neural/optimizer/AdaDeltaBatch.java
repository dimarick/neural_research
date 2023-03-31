package neural.optimizer;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import linear.MatrixF32;
import linear.Ops;
import neural.Layer;

import java.util.Arrays;

/**
 * Улучшение RMSProp: автоматическое управление скоростью
 */
public class AdaDeltaBatch extends BatchGradientDescent {
    private record MomentumItem(MatrixF32 m, MatrixF32 d, MatrixF32 g) {}
    private MomentumItem[] momentumData;
    private float alpha = 0.9f;
    private static final VectorSpecies<Float> species = FloatVector.SPECIES_MAX;

    public AdaDeltaBatch() {}

    public AdaDeltaBatch(float alpha) {
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
        float alpha1 = sgdItem.loss * layer.dropout.getRate();

        Ops.multiple(gradientMatrix, inputResult, momentum.g, alpha1, 0.0f);

        Ops.multipleBand(momentum.g.asVector(), momentum.g.asVector(), momentum.m.asVector(), 1.0f - alpha, alpha);

        var upperBound = species.loopBound(momentum.m.getSize());
        int length = species.length();
        var momentumData = momentum.m.getData();
        var deltaData = momentum.d.getData();
        var gradientData = momentum.g.getData();
        var outputData = layer.weights.getData();

        for (var j = 0; j < upperBound; j += length) {
            var m = FloatVector.fromArray(species, momentumData, j);
            var d = FloatVector.fromArray(species, deltaData, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            var o2 = g
                    .mul(d.sqrt().add(1e-10f))
                    .div(m.sqrt().add(1e-10f));

            o2.mul(-eta).add(w).intoArray(outputData, j);
            o2.mul(o2).mul(1 - alpha).add(d.mul(alpha)).intoArray(deltaData, j);
        }
    }

    protected void initStaticMemory(Layer[] layers, int batchSize) {
        momentumData = new MomentumItem[layers.length];
        super.initStaticMemory(layers, batchSize);
    }

    protected void initLayerStaticMemory(int i, Layer layer, int batchSize) {
        var w = layer.weights;
        if (w != null) {
            float[] oneMatrix = new float[w.getData().length];
            Arrays.fill(oneMatrix, 1f);
            momentumData[i] = new MomentumItem(
                    new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]),
                    new MatrixF32(w.getRows(), w.getColumns(), oneMatrix),
                    new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]));
        }
        super.initLayerStaticMemory(i, layer, batchSize);
    }
}
