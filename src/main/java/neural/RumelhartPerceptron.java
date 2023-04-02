package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class RumelhartPerceptron {

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    private final Random random;
    private final Optimizer.Interface optimizer;
    private final BackPropagation backPropagation = new BackPropagation();

    public RumelhartPerceptron(Random random, Optimizer.Interface optimizer) {
        this.random = random;
        this.optimizer = optimizer;
    }

    public int volume() {
        var v = 0;

        for (var layer : hiddenLayers) {
            v += layer.weights.getData().length;
        }

        return v + outputLayer.weights.getData().length;
    }

    public int inputSize() {
        return inputLayer.size;
    }

    public Layer addLayer(int size) {
        if (inputLayer == null) {
            inputLayer = new Layer(this, size, null);

            inputLayer.set(new Dropout.Zero(random, 0.0f));

            return inputLayer;
        }

        var previousLayer = outputLayer != null ? outputLayer : inputLayer;

        var layer = new Layer(this, size, new MatrixF32(previousLayer.size, size, new float[previousLayer.size * size], true));

        generateWeights(layer.weights.getData(), random, size);

        if (outputLayer != null) {
            hiddenLayers.add(outputLayer);
        }

        outputLayer = layer;

        layer.set(new Dropout.Zero(random, 0.0f));

        return layer;
    }

    public Layer addLayer(int size, Activation.Interface activation) {
        return addLayer(size).set(activation);
    }

    private void generateWeights(float[] layer, Random random, int size) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = (float)random.nextGaussian(0.0f, 1f / Math.sqrt(size));
        }
    }

    public float[] evalBatch(float[] sensorData) {
        if ((sensorData.length % inputLayer.size) != 0) {
            throw new RuntimeException();
        }

        var result = new MatrixF32(sensorData.length / inputLayer.size, inputLayer.size, sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayerBatch(result, layer);
        }

        result = evalLayerBatch(result, outputLayer);

        return result.getData();
    }

    public float[] trainBatch(float[] sensorData, float[] target, float eta) {
        if ((sensorData.length % inputLayer.size) != 0) {
            throw new RuntimeException();
        }

        var layerInput = new MatrixF32(sensorData.length / inputLayer.size, inputLayer.size, sensorData.clone());

        inputLayer.dropoutIndexes = inputLayer.dropout.init(layerInput.getSize());
        inputLayer.dropout.apply(layerInput, inputLayer.dropoutIndexes);

        var layerResult = new MatrixF32[hiddenLayers.size() + 2];
        var layers = new Layer[hiddenLayers.size() + 2];

        layerResult[0] = layerInput;
        layers[0] = inputLayer;

        for (int i = 0; i < hiddenLayers.size(); i++) {
            Layer layer = hiddenLayers.get(i);

            layerInput = evalLayerBatch(layerInput, layer);
            layer.dropoutIndexes = layer.dropout.init(layer.size);
            layer.dropout.apply(layerInput, layer.dropoutIndexes);

            layerResult[i + 1] = layerInput;
            layers[i + 1] = layer;
        }
        var result = evalLayerBatch(layerInput, outputLayer);
        outputLayer.dropoutIndexes = outputLayer.dropout.init(result.getSize());
        outputLayer.dropout.apply(result, outputLayer.dropoutIndexes);

        layerResult[hiddenLayers.size() + 1] = result;
        layers[hiddenLayers.size() + 1] = outputLayer;

        backPropagation.apply(optimizer, layers, layerResult, new MatrixF32(layerInput.getRows(), outputLayer.size, target), eta);

        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
            for (var i = hiddenLayers.size() - 1; i > 0; i--) {
                hiddenLayers.get(i).regularization.apply(hiddenLayers.get(i).weights, eta);
            }
            outputLayer.regularization.apply(outputLayer.weights, eta);
        }

        return result.getData();
    }

    private static MatrixF32 evalLayerBatch(MatrixF32 result, Layer layer) {
        float[] oneMat = new float[result.getRows()];
        Arrays.fill(oneMat, 1f);

        var r = Ops.multiple(result, layer.weights, 1.0f, 0.0f);
        Ops.multiple(new VectorF32(oneMat), layer.bias, r, 1f, 1f);

        r = layer.activation.applyBatch(r);

        return r;
    }
}
