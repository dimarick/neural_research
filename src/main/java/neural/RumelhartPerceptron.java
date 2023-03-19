package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.util.ArrayList;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class RumelhartPerceptron {

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    final private Random random;
    public Optimizer.Interface optimizer;
    public Optimizer.BatchInterface batchOptimizer;

    public RumelhartPerceptron(Random random, Optimizer.Interface optimizer) {
        this.random = random;
        this.optimizer = optimizer;
    }

    public RumelhartPerceptron(Random random, Optimizer.BatchInterface batchOptimizer) {
        this.random = random;
        this.batchOptimizer = batchOptimizer;
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

    public float[] eval(float[] sensorData) {
        var result = new VectorF32(sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayer(result, layer);
        }

        result = Ops.multiple(result, outputLayer.weights);
        result = outputLayer.activation.apply(result);

        return result.getData();
    }

    public float[] evalBatch(float[] sensorData) {
        if ((sensorData.length % inputLayer.size) != 0) {
            throw new RuntimeException();
        }

        var result = new MatrixF32(sensorData.length / inputLayer.size, inputLayer.size, sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayerBatch(result, layer);
        }

        result = Ops.multiple(result, outputLayer.weights);
        result = outputLayer.activation.applyBatch(result);

        return result.getData();
    }

    public float[] train(float[] sensorData, float[] target, float speed) {
        var layerInput = new VectorF32(sensorData.clone());

        inputLayer.dropoutIndexes = inputLayer.dropout.init(layerInput.getSize());
        inputLayer.dropout.apply(layerInput, inputLayer.dropoutIndexes);

        var layerResult = new VectorF32[hiddenLayers.size() + 2];
        var layers = new Layer[hiddenLayers.size() + 2];

        layerResult[0] = new VectorF32(sensorData);
        layers[0] = inputLayer;

        for (int i = 0; i < hiddenLayers.size(); i++) {
            Layer layer = hiddenLayers.get(i);

            layerInput = evalLayer(layerInput, layer);
            layer.dropoutIndexes = layer.dropout.init(layer.size);
            layer.dropout.apply(layerInput, layer.dropoutIndexes);

            layerResult[i + 1] = layerInput;
            layers[i + 1] = layer;
        }

        var result = Ops.multiple(layerInput, outputLayer.weights);
        result = outputLayer.activation.apply(result);
        outputLayer.dropoutIndexes = outputLayer.dropout.init(result.getSize());
        outputLayer.dropout.apply(result, outputLayer.dropoutIndexes);

        layerResult[hiddenLayers.size() + 1] = result;
        layers[hiddenLayers.size() + 1] = outputLayer;

        optimizer.apply(layers, layerResult, new VectorF32(target), speed);

        if (new Random().nextFloat(0.0f, 1.0f) > 0.99f) {
            for (var i = hiddenLayers.size() - 1; i > 0; i--) {
                hiddenLayers.get(i).regularization.apply(hiddenLayers.get(i).weights);
            }
            outputLayer.regularization.apply(outputLayer.weights);
        }

        return result.getData();
    }

    public float[] trainBatch(float[] sensorData, float[] target, float speed) {
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

        var result = Ops.multiple(layerInput, outputLayer.weights);
        result = outputLayer.activation.applyBatch(result);
        outputLayer.dropoutIndexes = outputLayer.dropout.init(result.getSize());
        outputLayer.dropout.apply(result, outputLayer.dropoutIndexes);

        layerResult[hiddenLayers.size() + 1] = result;
        layers[hiddenLayers.size() + 1] = outputLayer;

        batchOptimizer.apply(layers, layerResult, new MatrixF32(layerInput.getRows(), outputLayer.size, target), speed);

        if (new Random().nextFloat(0.0f, 1.0f) > 0.99f) {
            for (var i = hiddenLayers.size() - 1; i > 0; i--) {
                hiddenLayers.get(i).regularization.apply(hiddenLayers.get(i).weights);
            }
            outputLayer.regularization.apply(outputLayer.weights);
        }

        return result.getData();
    }

    private static VectorF32 evalLayer(VectorF32 result, Layer layer) {
        var r = Ops.multiple(result, layer.weights, 1.0f, 0.0f);

        r = layer.activation.apply(r);

        return r;
    }

    private static MatrixF32 evalLayerBatch(MatrixF32 result, Layer layer) {
        var r = Ops.multiple(result, layer.weights, 1.0f, 0.0f);

        r = layer.activation.applyBatch(r);

        return r;
    }
}
