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
public class AdamBatch extends BatchGradientDescent {
    private record MomentumItem(MatrixF32 g, MatrixF32 m, MatrixF32 v, MatrixF32 d, float[] beta) {}
    private MomentumItem[] momentumData;
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private static final VectorSpecies<Float> species = FloatVector.SPECIES_MAX;

    public AdamBatch() {}

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
        Ops.multipleBand(momentum.g.asVector(), momentum.d.asVector(), momentum.m.asVector(), 1 - beta1, beta1);
        Ops.multipleBand(momentum.g.asVector(), momentum.g.asVector(), momentum.v.asVector(), 1 - beta2, beta2);

        var upperBound = species.loopBound(momentum.v.getSize());
        int length = species.length();
        var momentumData = momentum.m.getData();
        var velocityData = momentum.v.getData();
        var gradientData = momentum.g.getData();
        var outputData = layer.weights.getData();

        for (var j = 0; j < upperBound; j += length) {
            var m = FloatVector.fromArray(species, momentumData, j);
            var v = FloatVector.fromArray(species, velocityData, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            var o =
                    m.div(1 - momentum.beta[0]).mul(-eta)
                    .div(v.div(1 - momentum.beta[1]).sqrt().add(1e-10f))
                    .add(w);

            o.intoArray(outputData, j);
        }

        momentum.beta[0] *= beta1;
        momentum.beta[1] *= beta2;
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
                    new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]),
                    new MatrixF32(w.getRows(), w.getColumns(), new float[w.getData().length]),
                    new MatrixF32(w.getRows(), w.getColumns(), oneMatrix),
                    new float[]{beta1, beta2}
            );
        }
        super.initLayerStaticMemory(i, layer, batchSize);
    }
}
