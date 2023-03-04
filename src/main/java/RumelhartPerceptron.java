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
    public static final float LOSS_THRESHOLD = 0.7f;

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

    private Layer getLastHiddenLayer() {
        return hiddenLayers.get(hiddenLayers.size() - 1);
    }

    public float[] train(float[] sensorData, float[] target, float speed, float dropoutFactor) {
        MatrixF32Interface hiddenResult = new MatrixF32(inputLayer.size, 1, sensorData);
        for (var layer : hiddenLayers) {
            hiddenResult = evalLayer(hiddenResult, layer);
        }

        if (dropoutFactor > 0) {
            dropout(hiddenResult.getData(), dropoutFactor);
        }

        var result = Ops.multiple(outputLayer.weights, hiddenResult);
        final var resultData = result.getData();
        Ops.softmax(result, ALPHA);

        var loss = loss(resultData, target, LOSS_THRESHOLD);

        var delta = new float[outputLayer.size];

        float alpha = speed * loss * (1.0f / (1 - dropoutFactor));

        for (var i = 0; i < outputLayer.size; i++) {
            delta[i] =  alpha * (target[i] - resultData[i]);
        }

        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
            float l1 = generalizeLasso();
//            float l1 = generalizeRidge();

            generalizationApply(l1);
        }

        Ops.multiple(new MatrixF32(outputLayer.size, 1, delta), Ops.transposeVector(hiddenResult), outputLayer.weights, 1.0f, 1.0f).getData();

        return resultData;
    }

    private static MatrixF32Interface evalLayer(MatrixF32Interface result, Layer layer) {
        result = Ops.multiple(layer.weights, result, 1.0f, 0.0f);
        Ops.reLU(result);
        Ops.normalize(result);
        return result;
    }

    private void generalizationApply(float l1) {
        var a = outputLayer.weights.getData();
        for (var i = 0; i < getLastHiddenLayer().size; i++) {
            a[i] = a[i] > 0 ? Math.max(0, a[i] - l1 * GENERALIZATION_FACTOR) : Math.min(0, a[i] + l1 * GENERALIZATION_FACTOR);
        }
    }

    private float loss(float[] result, float[] target, float threshold) {
        var a = getAnswer(target);

        var loss = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (a == i) {
                continue;
            }

            loss = Math.abs(Math.max(0, result[i] - result[a] + threshold));
        }

        return loss / result.length;
    }

    private void dropout(float[] result, float k) {
        for (var i : random.ints((long)(-result.length * Math.log(1 - k)), 0, result.length).toArray()) {
            result[i] = 0.0f;
        }
    }

    private static int getAnswer(float[] result) {
        var a = 0;
        var max = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (result[i] > max) {
                max = result[i];
                a = i;
            }
        }

        return a;
    }
    private float generalizeLasso() {
        float[] data = outputLayer.weights.getData();
        var result = 0.0f;

        for (float item : data) {
            result += Math.abs(item);
        }

        return result / data.length;
    }

    private float generalizeRidge() {
        float[] data = outputLayer.weights.getData();
        var result = 0.0f;

        for (float item : data) {
            result += item * item;
        }

        return result / data.length;
    }
}
