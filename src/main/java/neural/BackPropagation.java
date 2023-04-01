package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.util.Arrays;

final public class BackPropagation {
    private static class BpDataItem {
        protected final MatrixF32 diff;
        protected final VectorF32 error;
        protected final MatrixF32 errorMatrix;
        protected final VectorF32 inputGradient;
        protected final MatrixF32 weightsGradient;
        protected int i;

        protected BpDataItem(MatrixF32 diff, VectorF32 error, MatrixF32 errorMatrix, VectorF32 inputGradient, MatrixF32 weightsGradient, int i) {
            this.diff = diff;
            this.error = error;
            this.errorMatrix = errorMatrix;
            this.inputGradient = inputGradient;
            this.weightsGradient = weightsGradient;
            this.i = i;
        }
    }

    private BpDataItem[] data;

    public float apply(Optimizer.Interface2 optimizer, Layer[] layers, MatrixF32[] layerResults, MatrixF32 target, float eta) {
        if (data == null || (data[0].diff.getRows() != layerResults[0].getRows())) {
            initStaticMemory(layers, layerResults[0].getRows());
        }

        applyBackpropagation(layers, layerResults, target);
        calculateGradients();
        updateWeights(optimizer, layers, layerResults, eta);
        updateBias(layers, layerResults);

        return calculateTotalLoss(layers, layerResults);
    }
    private void updateBias(Layer[] layers, MatrixF32[] layerResults) {
        Arrays.stream(data).skip(1).forEach(bpItem -> updateLayerBias(layers, layerResults, bpItem));
    }

    private void updateLayerBias(Layer[] layers, MatrixF32[] layerResults, BpDataItem bpItem) {
        var i = bpItem.i;
        var layer = layers[i];
        MatrixF32 inputResult = layerResults[i - 1];
        var batchSize = inputResult.getRows();
        MatrixF32 gradientMatrix = new MatrixF32(batchSize, layer.size, bpItem.inputGradient.getData());
        float[] avgMatrix = new float[batchSize];
        Arrays.fill(avgMatrix, 1f / batchSize);
        Ops.multiple(new VectorF32(avgMatrix), gradientMatrix, layer.bias, -0.1f, 1.0f);
    }

    private float calculateTotalLoss(Layer[] layers, MatrixF32[] layerResults) {
        return (float) Arrays.stream(data).mapToDouble(m -> (double) layers[m.i].loss.apply(m.inputGradient, new VectorF32(layerResults[m.i].getData()))).sum();
    }

    private void updateWeights(Optimizer.Interface2 optimizer, Layer[] layers, MatrixF32[] layerResults, float eta) {
        Arrays.stream(data).skip(1).forEach(bpItem -> updateLayerWeights(optimizer, layers, layerResults, eta, bpItem));
    }

    private void updateLayerWeights(Optimizer.Interface2 optimizer, Layer[] layers, MatrixF32[] layerResults, float eta, BpDataItem bpItem) {
        var i = bpItem.i;
        var layer = layers[i];
        MatrixF32 inputResult = layerResults[i - 1];
        var batchSize = inputResult.getRows();
        MatrixF32 gradientMatrix = new MatrixF32(batchSize, layer.size, bpItem.inputGradient.getData()).transpose();
        Ops.multiple(gradientMatrix, inputResult, bpItem.weightsGradient, 1.0f, 1.0f);
        optimizer.apply(i, layer.weights.asVector(), bpItem.weightsGradient.asVector(), eta * layer.dropout.getRate());
    }

    private void calculateGradients() {
        Arrays.stream(data).skip(1).parallel().forEach(mem -> {
            Ops.multipleBand(mem.error, new VectorF32(mem.diff.getData()), mem.inputGradient, 1.0f, 0.0f);
        });
    }

    private void applyBackpropagation(Layer[] layers, MatrixF32[] layerResults, MatrixF32 target) {
        int outLayerId = layerResults.length - 1;
        var result = layerResults[outLayerId];
        var outputLayer = layers[outLayerId];

        var outMemory = data[outLayerId];

        outMemory.i = data.length - 1;
        outputLayer.activation.diffBatch(result, outMemory.diff);
        outputLayer.dropout.apply(outMemory.diff, outputLayer.dropoutIndexes);

        System.arraycopy(result.getData(), 0, outMemory.error.getData(), 0, target.getData().length);
        Ops.add(target.getData(), outMemory.error.getData(), -1.0f);

        for (var i = layers.length - 2; i > 0; i--) {
            var layer = layers[i];
            var mem = data[i];

            mem.i = i;
            layer.activation.diffBatch(layerResults[i], mem.diff);
            layer.dropout.apply(mem.diff, layer.dropoutIndexes);
            Ops.multiple(data[i + 1].errorMatrix, layers[i + 1].weights.transpose(), mem.errorMatrix, 1.0f, 0.0f).getData();
        }
    }

    private void initStaticMemory(Layer[] layers, int batchSize) {
        data = new BpDataItem[layers.length];
        for (var i = 0; i < layers.length; i++) {
            initLayerStaticMemory(i, layers[i], batchSize);
        }
    }

    private void initLayerStaticMemory(int i, Layer layer, int batchSize) {
        var size = layer.size * batchSize;
        var err = new float[size];
        data[i] = new BpDataItem(
            new MatrixF32(batchSize, layer.size, new float[size]),
            new VectorF32(err),
            new MatrixF32(batchSize, layer.size, err),
            new VectorF32(new float[size]),
            new MatrixF32(layer.weights.getRows(), layer.weights.getColumns()),
            0
        );
    }
}
