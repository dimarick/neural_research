package neural.optimizer;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;
import neural.Layer;
import neural.Optimizer;

import java.util.Arrays;

public class BatchGradientDescent implements Optimizer.BatchInterface {
    protected static class SgdDataItem {
        protected final MatrixF32 diff;
        protected final VectorF32 error;
        protected final MatrixF32 errorMatrix;
        protected final VectorF32 gradient;
        protected float loss;
        protected int i;

        protected SgdDataItem(MatrixF32 diff, VectorF32 error, MatrixF32 errorMatrix, VectorF32 gradient, float loss, int i) {
            this.diff = diff;
            this.error = error;
            this.errorMatrix = errorMatrix;
            this.gradient = gradient;
            this.loss = loss;
            this.i = i;
        }
    }

    protected SgdDataItem[] sgdData;

    public float apply(Layer[] layers, MatrixF32[] layerResults, MatrixF32 target, float eta) {
        if (sgdData == null) {
            initStaticMemory(layers, layerResults[0].getRows());
        }

        applyBackpropagation(layers, layerResults, target);
        calculateGradients(layers, layerResults);
        updateWeights(layers, layerResults, eta);
        updateBias(layers, layerResults, eta);

        return calculateTotalLoss();
    }
    private void updateBias(Layer[] layers, MatrixF32[] layerResults, float eta) {
        Arrays.stream(sgdData).skip(1).forEach(sgdItem -> updateLayerBias(layers, layerResults, eta, sgdItem));
    }


    private void updateLayerBias(Layer[] layers, MatrixF32[] layerResults, float eta, SgdDataItem sgdItem) {
        var i = sgdItem.i;
        var layer = layers[i];
        MatrixF32 inputResult = layerResults[i - 1];
        var batchSize = inputResult.getRows();
        MatrixF32 gradientMatrix = new MatrixF32(batchSize, layer.size, sgdItem.gradient.getData());
        float[] avgMatrix = new float[batchSize];
        Arrays.fill(avgMatrix, 1f / batchSize);
        Ops.multiple(new VectorF32(avgMatrix), gradientMatrix, layer.bias, -0.1f, 1.0f);
    }

    protected float calculateTotalLoss() {
        return (float) Arrays.stream(sgdData).mapToDouble(m -> (double) m.loss).sum();
    }

    protected void updateWeights(Layer[] layers, MatrixF32[] layerResults, float eta) {
        Arrays.stream(sgdData).skip(1).forEach(sgdItem -> updateLayerWeights(layers, layerResults, eta, sgdItem));
    }

    protected void updateLayerWeights(Layer[] layers, MatrixF32[] layerResults, float eta, SgdDataItem sgdItem) {
        var i = sgdItem.i;
        var layer = layers[i];
        MatrixF32 inputResult = layerResults[i - 1];
        var batchSize = inputResult.getRows();
        MatrixF32 gradientMatrix = new MatrixF32(batchSize, layer.size, sgdItem.gradient.getData()).transpose();
        Ops.multiple(gradientMatrix, inputResult, layer.weights, -eta * sgdItem.loss * layer.dropout.getRate(), 1.0f);
    }

    protected void calculateGradients(Layer[] layers, MatrixF32[] layerResults) {
        Arrays.stream(sgdData).skip(1).parallel().forEach(mem -> {
            var i = mem.i;
            var layer = layers[i];
            Ops.multipleBand(mem.error, new VectorF32(mem.diff.getData()), mem.gradient, 1.0f, 0.0f);
            mem.loss = 1f;
        });
    }

    protected void applyBackpropagation(Layer[] layers, MatrixF32[] layerResults, MatrixF32 target) {
        int outLayerId = layerResults.length - 1;
        var result = layerResults[outLayerId];
        var outputLayer = layers[outLayerId];

        var outMemory = sgdData[outLayerId];

        outMemory.i = sgdData.length - 1;
        outputLayer.activation.diffBatch(result, outMemory.diff);
        outputLayer.dropout.apply(outMemory.diff, outputLayer.dropoutIndexes);

        System.arraycopy(result.getData(), 0, outMemory.error.getData(), 0, target.getData().length);
        Ops.add(target.getData(), outMemory.error.getData(), -1.0f);

        for (var i = layers.length - 2; i > 0; i--) {
            var layer = layers[i];
            var mem = sgdData[i];

            mem.i = i;
            layer.activation.diffBatch(layerResults[i], mem.diff);
            layer.dropout.apply(mem.diff, layer.dropoutIndexes);
            Ops.multiple(sgdData[i + 1].errorMatrix, layers[i + 1].weights.transpose(), mem.errorMatrix, 1.0f, 0.0f).getData();
        }
    }

    protected void initStaticMemory(Layer[] layers, int batchSize) {
        sgdData = new SgdDataItem[layers.length];
        for (var i = 0; i < layers.length; i++) {
            initLayerStaticMemory(i, layers[i], batchSize);
        }
    }

    protected void initLayerStaticMemory(int i, Layer layer, int batchSize) {
        var size = layer.size * batchSize;
        var err = new float[size];
        sgdData[i] = new SgdDataItem(new MatrixF32(batchSize, layer.size, new float[size]), new VectorF32(err), new MatrixF32(batchSize, layer.size, err), new VectorF32(new float[size]), 0, 0);
    }
}