package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Реализация многослойного перцептрона с обратным распространением ошибки
 */
public class FeedForwardNeuralNetwork {

    final private ArrayList<Layer> hiddenLayers = new ArrayList<>();
    private Layer inputLayer;
    private Layer outputLayer;
    private final Random random;
    private final Optimizer.Interface optimizer;
    private final BackPropagation backPropagation = new BackPropagation();

    public FeedForwardNeuralNetwork(Random random, Optimizer.Interface optimizer) {
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

    public void dumpLayersStat(PrintStream out) {
        for (var layer : hiddenLayers) {
            dumpLayerStat(out, layer);
        }

        dumpLayerStat(out, outputLayer);
    }

    private void dumpLayerStat(PrintStream out, Layer layer) {
        out.println("Layer " + layer.size);
        if (layer.weights != null) {
            var min = Float.POSITIVE_INFINITY;
            var max = Float.NEGATIVE_INFINITY;
            var mean = 0f;
            var disp = 0f;
            var histogram = new int[10];
            var histogramLegend = new double[10];

            for (var w : layer.weights.getData()) {
                min = Math.min(min, w);
                max = Math.max(max, w);
                mean += w / (double)layer.weights.getSize();
                disp += w * w;
            }

            var histStep = (max - min) / histogram.length;

            for (var w : layer.weights.getData()) {
                var v = Math.min(histogram.length - 1, (int)Math.floor((w - min) / histStep));
                histogram[v]++;
                histogramLegend[v] = histStep * v + min;
            }

            out.println("\t w: k=" + layer.weights.getSize() + "\tmin: " + min + "\t max: " + max + "\tmean: " + mean + "\tstddev: " + Math.sqrt(disp));

            out.print("\t");

            for (var i = 0; i < histogram.length; i++) {
                out.print("\t" + (double)Math.round(histogramLegend[i] * 1000) / 1000);
            }

            out.println();
            out.print("\t");

            for (var i = 0; i < histogram.length; i++) {
                out.print("\t" + histogram[i]);
            }

            out.println();
        }
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
            var v = (float)Math.sqrt(size);
            layer[i] = random.nextFloat(-0.5f / v, 0.5f / v);
//            layer[i] = (float)random.nextGaussian(0.0f, 1f / v);
        }
    }

    public float[] eval(float[] sensorData) {
        if ((sensorData.length % inputLayer.size) != 0) {
            throw new RuntimeException();
        }

        var result = new MatrixF32(sensorData.length / inputLayer.size, inputLayer.size, sensorData);

        for (var layer : hiddenLayers) {
            result = evalLayer(result, layer);
        }

        result = evalLayer(result, outputLayer);

        return result.getData();
    }

    public float[] train(float[] sensorData, float[] target, float eta) {
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

            layerInput = evalLayer(layerInput, layer);
            layer.dropoutIndexes = layer.dropout.init(layer.size);
            layer.dropout.apply(layerInput, layer.dropoutIndexes);

            layerResult[i + 1] = layerInput;
            layers[i + 1] = layer;
        }
        var result = evalLayer(layerInput, outputLayer);
        outputLayer.dropoutIndexes = outputLayer.dropout.init(result.getSize());
        outputLayer.dropout.apply(result, outputLayer.dropoutIndexes);

        layerResult[hiddenLayers.size() + 1] = result;
        layers[hiddenLayers.size() + 1] = outputLayer;

        backPropagation.apply(optimizer, layers, layerResult, new MatrixF32(layerInput.getRows(), outputLayer.size, target), eta);
//
//        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
//            for (var i = hiddenLayers.size() - 1; i > 0; i--) {
//                hiddenLayers.get(i).regularization.apply(hiddenLayers.get(i).weights, 1);
//            }
//            outputLayer.regularization.apply(outputLayer.weights, 1);
//        }

        return result.getData();
    }

    private static MatrixF32 evalLayer(MatrixF32 result, Layer layer) {
        float[] I = new float[result.getRows()];
        Arrays.fill(I, 1f);

        var r = Ops.product(result, layer.weights, 1.0f, 0.0f);
        Ops.product(new VectorF32(I), layer.bias, r, 1f, 1f);

        r = layer.activation.applyBatch(r);

        return r;
    }
}
