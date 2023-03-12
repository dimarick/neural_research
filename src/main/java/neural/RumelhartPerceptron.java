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

    public RumelhartPerceptron(Random random, Optimizer.Interface optimizer) {
        this.random = random;
        this.optimizer = optimizer;
    }

    public Layer addLayer(int size) {
        if (inputLayer == null) {
            inputLayer = new Layer(this, size, null);

            inputLayer.set(new Dropout.Zero(random, 0.0f));

            return inputLayer;
        }

        var previousLayer = outputLayer != null ? outputLayer : inputLayer;

        var layer = new Layer(this, size, new MatrixF32(size, previousLayer.size));

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
            layer[i] = (float)random.nextGaussian(0.0f, 1f / size);
        }
    }

    public float[] eval(float[] sensorData) {
        var result = new VectorF32(sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayer(result, layer);
        }

        result = Ops.multiple(outputLayer.weights, result);
        result = outputLayer.activation.apply(result);

        return result.getData();
    }

    public float[] train(float[] sensorData, float[] target, float speed) {
        var layerInput = new VectorF32(sensorData);
        var layerResult = new VectorF32[hiddenLayers.size() + 2];
        var layers = new Layer[hiddenLayers.size() + 2];

        layerResult[0] = new VectorF32(sensorData);
        layers[0] = inputLayer;

        for (int i = 0; i < hiddenLayers.size(); i++) {
            Layer layer = hiddenLayers.get(i);

            layerInput = evalLayer(layerInput, layer);

            layerResult[i + 1] = layerInput;
            layers[i + 1] = layer;
        }

        var result = Ops.multiple(outputLayer.weights, layerInput);
        result = outputLayer.activation.apply(result);

        layerResult[hiddenLayers.size() + 1] = result;
        layers[hiddenLayers.size() + 1] = outputLayer;

        optimizer.apply(layers, layerResult, new VectorF32(target), speed);

        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
            for (var i = hiddenLayers.size() - 1; i > 0; i--) {
                hiddenLayers.get(i).regularization.apply(hiddenLayers.get(i).weights);
            }
            outputLayer.regularization.apply(outputLayer.weights);
        }

        return result.getData();
    }

    private static VectorF32 evalLayer(VectorF32 result, Layer layer) {
        var r = Ops.multiple(layer.weights, result, 1.0f, 0.0f);

        r = layer.activation.apply(r);
        layer.dropout.apply(r);

        return r;
    }
}
