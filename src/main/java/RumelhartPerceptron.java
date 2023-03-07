import linear.matrix.MatrixF32;
import linear.matrix.Ops;
import neural.NeuralAlgo;

import java.util.ArrayList;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class RumelhartPerceptron {
    public interface ActivationFunction {
        void apply(MatrixF32 value);
    }

    private static class Layer {
        final private int size;
        final private MatrixF32 weights;
        final private ActivationFunction activationFunction;

        public Layer(int size, MatrixF32 weights, ActivationFunction activationFunction) {
            this.size = size;
            this.weights = weights;
            this.activationFunction = activationFunction;
        }
    }


    public static final float ALPHA = 1.0f;
    public static final float GENERALIZATION_FACTOR = 1e-4f;

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    final private Random random;
    public RumelhartPerceptron(Random random) {
        this.random = random;
    }

    public RumelhartPerceptron addLayer(int size, ActivationFunction activationFunction) {
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
        return addLayer(size, r -> {
            NeuralAlgo.softmax(r, ALPHA);
        });
    }

    private void generateWeights(float[] layer, Random random, int size) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = (float)random.nextGaussian(0.0f, 1f / size);
        }
    }

    public float[] eval(float[] sensorData) {
        MatrixF32 result = new MatrixF32(inputLayer.size, 1, sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayer(result, layer);
        }

        result = Ops.multiple(outputLayer.weights, result);
        NeuralAlgo.normalize(result);
        NeuralAlgo.softmax(result, ALPHA);

        return result.getData();
    }

    public float[] train(float[] sensorData, float[] target, float speed, float dropoutFactor) {
        MatrixF32 hiddenResult = new MatrixF32(inputLayer.size, 1, sensorData);
        MatrixF32[] hiddenResults = new MatrixF32[hiddenLayers.size()];
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
        outputLayer.activationFunction.apply(result);

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

            var layerError = new float[currentResult.getData().length];

            var currentResultDiff = NeuralAlgo.reLUDiff(currentResult);

            for (var j = 0; j < layerError.length; j++) {
                var sum = 0.0f;
                for (var k = 0; k < nextLayerResult.length; k++) {
                    sum += nextLayerLoss[k] * nextLayer[k * currentResult.getData().length + j];
                }

                layerError[j] = currentResultDiff[j] * sum;
            }

            var w = hiddenLayers.get(i).weights.getData();

            int size = i > 0 ? hiddenLayers.get(i - 1).size : inputLayer.size;
            for (var j = 0; j < layerError.length; j++) {
                for (var l = 0; l < size; l++) {
                    w[j * size + l] += -speed * loss * hiddenResults[i].getData()[l] * layerError[j] / Math.sqrt(size);
                }
            }

            nextLayerResult = currentResult.getData();
            nextLayer = hiddenLayers.get(i).weights.getData();
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

    private static MatrixF32 evalLayer(MatrixF32 result, Layer layer) {
        var r = Ops.multiple(layer.weights, result, 1.0f, 0.0f);

        layer.activationFunction.apply(r);

        return r;
    }
}
