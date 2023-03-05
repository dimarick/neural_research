import linear.matrix.MatrixF32;
import linear.matrix.MatrixF32Interface;
import linear.matrix.Ops;

import java.util.ArrayList;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class RumelhartPerceptron {
    private class Layer {
        final private int size;
        final private MatrixF32 weights;

        public Layer(int size, MatrixF32 weights) {
            this.size = size;
            this.weights = weights;
        }
    }


    public static final float ALPHA = 1.0f;
    public static final float GENERALIZATION_FACTOR = 1e-6f;

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    final private Random random;
    public RumelhartPerceptron(Random random) {
        this.random = random;
    }

    public RumelhartPerceptron addLayer(int size) {
        if (inputLayer == null) {
            inputLayer = new Layer(size, null);

            return this;
        }

        var previousLayer = outputLayer != null ? outputLayer : inputLayer;

        var layer = new Layer(size, new MatrixF32(size, previousLayer.size));

        generateWeights(layer.weights.getData(), random);

        if (outputLayer != null) {
            hiddenLayers.add(outputLayer);
        }

        outputLayer = layer;

        return this;
    }

    private void generateWeights(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = (float)random.nextGaussian(0.0f, 1f);
        }
    }

    public float[] eval(float[] sensorData) {
        MatrixF32Interface result = new MatrixF32(inputLayer.size, 1, sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayer(result, layer);
        }

        result = Ops.multiple(outputLayer.weights, result);
        Ops.normalize(result);
        Ops.softmax(result, ALPHA);

        return result.getData();
    }

    public float[] train(float[] sensorData, float[] target, float speed, float dropoutFactor) {
        MatrixF32Interface hiddenResult = new MatrixF32(inputLayer.size, 1, sensorData);
        for (var layer : hiddenLayers) {
            hiddenResult = evalLayer(hiddenResult, layer);
        }

        float[] hiddenData = hiddenResult.getData();
        if (dropoutFactor > 0) {
            Ops.dropout(random, hiddenData, dropoutFactor);
        }

        var result = Ops.multiple(outputLayer.weights, hiddenResult);
        final var resultData = result.getData();
        Ops.softmax(result, ALPHA);
        var diff = Ops.softmaxDiff(result, ALPHA);
        var delta = new float[outputLayer.size];

        float alpha = speed * Ops.dropoutRate(dropoutFactor);

        for (var i = 0; i < outputLayer.size; i++) {
            delta[i] = -alpha * (result.getData()[i] - target[i]) * diff[i];
        }

        Ops.multiple(new MatrixF32(outputLayer.size, 1, delta), Ops.transposeVector(hiddenResult), outputLayer.weights, 1.0f, 1.0f).getData();

        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
            float l1 = Ops.generalizeLasso(outputLayer.weights);
//            float l1 = Ops.generalizeRidge(outputLayer.weights);

            Ops.generalizationApply(l1, outputLayer.weights, GENERALIZATION_FACTOR);
        }

        return resultData;
    }

    private static MatrixF32Interface evalLayer(MatrixF32Interface result, Layer layer) {
        result = Ops.multiple(layer.weights, result, 1.0f, 0.0f);
        Ops.reLU(result);
        Ops.normalize(result);
        return result;
    }
}
