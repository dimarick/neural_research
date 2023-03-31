package neural.optimizer;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import linear.MatrixF32;
import linear.Ops;
import neural.Layer;

/**
 * Улучшение AdaGrad: вместо полного среднего используем экспоненциальное скользящее среднее
 * Решает проблему снижения скорости обучения до нуля раньше чем выборка будет разобрана
 * Наглядно очевиден рост скорости схождения в разы: 17 эпох на 160 нейронов
 */
public class RMSPropBatch extends BatchGradientDescent {
    private record MomentumItem(MatrixF32 m, MatrixF32 g) {}
    private MomentumItem[] momentumData;
    private float alpha = 0.9f;
    private static final VectorSpecies<Float> species = FloatVector.SPECIES_MAX;

    public RMSPropBatch() {}

    public RMSPropBatch(float alpha) {
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
        var gradientData = momentum.g.getData();
        var outputData = layer.weights.getData();
//
//        NeuralAlgo.parallel((t, cores) -> {
//            var rangeSize = (int) Math.ceil((double) upperBound / cores / length) * length;
//            var start = t * rangeSize;
//            var limit = Math.min(upperBound, t * rangeSize + rangeSize);
//            for (var j = start; j < limit; j += length) {
//                var m = FloatVector.fromArray(species, momentumData, j);
//                var g = FloatVector.fromArray(species, gradientData, j);
//                var w = FloatVector.fromArray(species, outputData, j);
//
//                var o2 = g.div(m.add(1e-5f).sqrt()).mul(-eta).add(w);
//
//                o2.intoArray(outputData, j);
//            }
//        }, 6);

        for (var j = 0; j < upperBound; j += length) {
            var m = FloatVector.fromArray(species, momentumData, j);
            var g = FloatVector.fromArray(species, gradientData, j);
            var w = FloatVector.fromArray(species, outputData, j);

            var o2 = g.div(m.add(1e-10f).sqrt()).mul(-eta).add(w);

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
