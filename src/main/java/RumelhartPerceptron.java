import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;
import neural.*;

import java.util.ArrayList;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class RumelhartPerceptron {
    public static final class Layer {
        private final RumelhartPerceptron parent;
        private final int size;
        private final MatrixF32 weights;
        private Activation.Interface activation;
        private Loss.Interface loss;
        private Generalization.Interface generalization;
        private Dropout.Interface dropout;

        private Layer(RumelhartPerceptron parent, int size, MatrixF32 weights) {
            this.parent = parent;
            this.size = size;
            this.weights = weights;
            this
                    .set(new Activation.Softmax())
                    .set(new Generalization.Lasso());
        }

        public RumelhartPerceptron parent() {
            return parent;
        }

        public Layer set(Activation.Interface activation) {
            this.activation = activation;

            return this;
        }

        public Layer set(Loss.Interface loss) {
            this.loss = loss;

            return this;
        }

        public Layer set(Generalization.Interface generalization) {
            this.generalization = generalization;

            return this;
        }

        public Layer set(Dropout.Interface dropout) {
            this.dropout = dropout;

            return this;
        }

        private Loss.Interface loss() {
            return loss != null ? loss : activation.suggestLoss();
        }
    }

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    final private Random random;
    public RumelhartPerceptron(Random random) {
        this.random = random;
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
        var hiddenResult = new VectorF32(sensorData);
        var hiddenResults = new VectorF32[hiddenLayers.size()];
        for (int i = 0; i < hiddenLayers.size(); i++) {
            hiddenResults[i] = hiddenResult;
            Layer layer = hiddenLayers.get(i);

            layer.dropout.apply(hiddenResult);

            hiddenResult = evalLayer(hiddenResult, layer);
        }

        outputLayer.dropout.apply(hiddenResult);

        var result = Ops.multiple(outputLayer.weights, hiddenResult);
        result = outputLayer.activation.apply(result);

        final var resultData = result.getData();

        var diff = outputLayer.activation.diff(result);
        var error = Ops.subtract(result.getData(), target);

        var loss = outputLayer.loss().apply(new VectorF32(target), result);

        if (loss == 0) {
            return resultData;
        }

        var nextLayerResult = result.getData();
        var nextLayer = outputLayer.weights.getData();
        var nextLayerLoss = error;
        var currentResult = hiddenResult;

        for (var i = hiddenLayers.size() - 1; i >= 0; i--) {
            var layer = hiddenLayers.get(i);

            float[] currentResultData = currentResult.getData();
            var layerTarget = currentResultData.clone();
            var layerError = currentResultData.clone();

            var currentResultDiff = layer.activation.diff(currentResult).getData();

            for (var j = 0; j < layerTarget.length; j++) {
                var sum = 0.0f;
                for (var k = 0; k < nextLayerResult.length; k++) {
                    sum += nextLayerLoss[k] * nextLayer[k * currentResultData.length + j];
                }

                layerTarget[j] += currentResultDiff[j] * sum;
                layerError[j] = currentResultDiff[j] * sum;
            }

            var loss2 = layer.loss().apply(new VectorF32(layerTarget), currentResult);

            Ops.multiple(new VectorF32(layerError), hiddenResults[i], layer.weights, -speed * loss2, 1.0f);

            nextLayerResult = currentResultData;
            nextLayer = layer.weights.getData();
            nextLayerLoss = layerError;
            currentResult = hiddenResults[i];
        }

        NeuralAlgo.sdg(
                speed * loss * outputLayer.dropout.getRate(hiddenResult),
                diff.getData(),
                error,
                hiddenResult,
                outputLayer.weights
        );


        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
            for (var i = hiddenLayers.size() - 1; i > 0; i--) {
                hiddenLayers.get(i).generalization.apply(hiddenLayers.get(i).weights);
            }
            outputLayer.generalization.apply(outputLayer.weights);
        }

        return resultData;
    }

    private static VectorF32 evalLayer(VectorF32 result, Layer layer) {
        var r = Ops.multiple(layer.weights, result, 1.0f, 0.0f);

        return layer.activation.apply(r);
    }
}
