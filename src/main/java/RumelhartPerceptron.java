import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;
import neural.Activation;
import neural.NeuralAlgo;

import java.util.ArrayList;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class RumelhartPerceptron {
    private record Layer(int size, MatrixF32 weights, Activation.Interface activationFunction) {}

    public static final float ALPHA = 1.0f;
    public static final float GENERALIZATION_FACTOR = 1e-4f;

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    final private Random random;
    public RumelhartPerceptron(Random random) {
        this.random = random;
    }

    public RumelhartPerceptron addLayer(int size, Activation.Interface activationFunction) {
        if (inputLayer == null) {
            inputLayer = new Layer(size, null, activationFunction);

            return this;
        }

        var previousLayer = outputLayer != null ? outputLayer : inputLayer;

        var layer = new Layer(size, new MatrixF32(size, previousLayer.size), activationFunction);

        generateWeights(layer.weights.getData(), random, size);

        if (outputLayer != null) {
            hiddenLayers.add(outputLayer);
        }

        outputLayer = layer;

        return this;
    }

    public RumelhartPerceptron addLayer(int size) {
        return addLayer(size, new Activation.Softmax(ALPHA));
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
        result = outputLayer.activationFunction.apply(result);

        return result.getData();
    }

    public float[] train(float[] sensorData, float[] target, float speed, float dropoutFactor) {
        var hiddenResult = new VectorF32(sensorData);
        var hiddenResults = new VectorF32[hiddenLayers.size()];
        for (int i = 0; i < hiddenLayers.size(); i++) {
            hiddenResults[i] = hiddenResult;
            Layer layer = hiddenLayers.get(i);
            hiddenResult = evalLayer(hiddenResult, layer);
        }

        float[] hiddenData = hiddenResult.getData();
        if (dropoutFactor > 0) {
            NeuralAlgo.dropout(random, hiddenData, dropoutFactor);
        }

        var result = Ops.multiple(outputLayer.weights, hiddenResult);
        result = outputLayer.activationFunction.apply(result);

        final var resultData = result.getData();
        var loss= NeuralAlgo.loss(resultData, target, 0.9f);

        if (loss == 0) {
            return resultData;
        }

        var diff = NeuralAlgo.softmaxDiff(result, ALPHA);
        var error = Ops.subtract(result.getData(), target);

        var nextLayerResult = result.getData();
        var nextLayer = outputLayer.weights.getData();
        var nextLayerLoss = error;
        var currentResult = hiddenResult;

        for (var i = hiddenLayers.size() - 1; i >= 0; i--) {
            var layer = hiddenLayers.get(i);

            var layerError = new float[currentResult.getData().length];

            var currentResultDiff = layer.activationFunction.diff(currentResult).getData();

            for (var j = 0; j < layerError.length; j++) {
                var sum = 0.0f;
                for (var k = 0; k < nextLayerResult.length; k++) {
                    sum += nextLayerLoss[k] * nextLayer[k * currentResult.getData().length + j];
                }

                layerError[j] = currentResultDiff[j] * sum;
            }

            int size = i > 0 ? hiddenLayers.get(i - 1).size : inputLayer.size;

            Ops.multiple(new VectorF32(layerError), hiddenResults[i], layer.weights, -speed * loss / (float)Math.sqrt(size), 1.0f);

            nextLayerResult = currentResult.getData();
            nextLayer = layer.weights.getData();
            nextLayerLoss = layerError;
            currentResult = hiddenResults[i];
        }

        NeuralAlgo.sdg(
                speed * loss * NeuralAlgo.dropoutRate(dropoutFactor),
                diff,
                error,
                hiddenResult,
                outputLayer.weights
        );


        for (var i = hiddenLayers.size() - 1; i > 0; i--) {

            if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
                float l1 = NeuralAlgo.generalizeLasso(hiddenLayers.get(i).weights);
//                float l1 = NeuralAlgo.generalizeRidge(hiddenLayers.get(i).weights);

                NeuralAlgo.generalizationApply(l1, hiddenLayers.get(i).weights, GENERALIZATION_FACTOR);
            }
        }

        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
            float l1 = NeuralAlgo.generalizeLasso(outputLayer.weights);
//            float l1 = NeuralAlgo.generalizeRidge(outputLayer.weights);

            NeuralAlgo.generalizationApply(l1, outputLayer.weights, GENERALIZATION_FACTOR);
        }

        return resultData;
    }

    private static VectorF32 evalLayer(VectorF32 result, Layer layer) {
        var r = Ops.multiple(layer.weights, result, 1.0f, 0.0f);

        return layer.activationFunction.apply(r);
    }
}
