package neural.optimizer;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;
import neural.Layer;
import neural.Optimizer;

import java.util.Arrays;

public class StochasticGradientDescent implements Optimizer.Interface {
    protected static class SgdDataItem {
        protected final VectorF32 diff;
        protected final VectorF32 error;
        protected final MatrixF32 errorMatrix;
        protected final VectorF32 gradient;
        protected float loss;
        protected int i;

        protected SgdDataItem(VectorF32 diff, VectorF32 error, MatrixF32 errorMatrix, VectorF32 gradient, float loss, int i) {
            this.diff = diff;
            this.error = error;
            this.errorMatrix = errorMatrix;
            this.gradient = gradient;
            this.loss = loss;
            this.i = i;
        }
    }

    protected SgdDataItem[] sgdData;

    public float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta) {
        if (sgdData == null) {
            initStaticMemory(layers);
        }

        applyBackpropagation(layers, layerResults, target);
        calculateGradients(layers, layerResults);
        updateWeights(layers, layerResults, eta);

        return calculateTotalLoss();
    }

    protected float calculateTotalLoss() {
        return (float) Arrays.stream(sgdData).mapToDouble(m -> (double) m.loss).sum();
    }

    protected void updateWeights(Layer[] layers, VectorF32[] layerResults, float eta) {
        Arrays.stream(sgdData).skip(1).forEach(sgdItem -> updateLayerWeights(layers, layerResults, eta, sgdItem));
    }

    protected void updateLayerWeights(Layer[] layers, VectorF32[] layerResults, float eta, SgdDataItem sgdItem) {
        var i = sgdItem.i;
        var layer = layers[i];
        Ops.multiple(sgdItem.gradient, layerResults[i - 1], layer.weights, -eta * sgdItem.loss * layer.dropout.getRate(), 1.0f);
    }

    protected void calculateGradients(Layer[] layers, VectorF32[] layerResults) {
        Arrays.stream(sgdData).skip(1).forEach(mem -> {
            var i = mem.i;
            var layer = layers[i];
            Ops.multipleBand(mem.error, mem.diff, mem.gradient, 1.0f, 0.0f);

            var layerLoss = layer.loss.apply(mem.gradient, layerResults[i]);
            // fast gradient clipping
            mem.loss = Math.signum(layerLoss) * Math.min(2, Math.abs(layerLoss));
        });
    }

    protected void applyBackpropagation(Layer[] layers, VectorF32[] layerResults, VectorF32 target) {
        int outLayerId = layerResults.length - 1;
        var result = layerResults[outLayerId];
        var outputLayer = layers[outLayerId];

        var outMemory = sgdData[outLayerId];

        outMemory.i = sgdData.length - 1;
        outputLayer.activation.diff(result, outMemory.diff);
        outputLayer.dropout.apply(outMemory.diff, outputLayer.dropoutIndexes);

        System.arraycopy(result.getData(), 0, outMemory.error.getData(), 0, target.getData().length);
        Ops.add(target.getData(), outMemory.error.getData(), -1.0f);

        for (var i = layers.length - 2; i > 0; i--) {
            var layer = layers[i];
            var mem = sgdData[i];

            mem.i = i;
            layer.activation.diff(layerResults[i], mem.diff).getData();
            layer.dropout.apply(mem.diff, layer.dropoutIndexes);
            Ops.multiple(sgdData[i + 1].errorMatrix, layers[i + 1].weights.transpose(), mem.errorMatrix, 1.0f, 0.0f).getData();
        }
    }

    protected void initStaticMemory(Layer[] layers) {
        sgdData = new SgdDataItem[layers.length];
        for (var i = 0; i < layers.length; i++) {
            initLayerStaticMemory(i, layers[i]);
        }
    }

    protected void initLayerStaticMemory(int i, Layer layer) {
        var size = layer.size;
        var err = new float[size];
        sgdData[i] = new SgdDataItem(new VectorF32(new float[size]), new VectorF32(err), new MatrixF32(1, err.length, err), new VectorF32(new float[size]), 0, 0);
    }
}