import linear.matrix.MatrixF32;
import linear.matrix.MatrixF32Interface;
import linear.matrix.Ops;

import java.util.Random;

public class RosenblattPerceptron {
    final private int outputLayerSize;
    final private int assocLayerSize;
    final private int maxThreads;
    final private MatrixF32 sensorLayer;
    final private MatrixF32 assocLayer;

    public RosenblattPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random, int maxThreads) {
        this.outputLayerSize = outputLayerSize;
        this.assocLayerSize = assocLayerSize;
        this.maxThreads = maxThreads;

        this.sensorLayer = new MatrixF32(assocLayerSize, sensorLayerSize);
        this.assocLayer = new MatrixF32(outputLayerSize, assocLayerSize);

        generateWeightsS(this.sensorLayer.getData(), random);
        generateWeightsA(this.assocLayer.getData(), random);
    }

    public float[] eval(float[] sensorData) {
        final var hiddenResultMatrix = Ops.multipleTransposedConcurrent(sensorLayer, new MatrixF32(1, sensorData.length, sensorData), maxThreads);

        final var hiddenResult = ((MatrixF32Interface) hiddenResultMatrix).getData();

        for (var i = 0; i < hiddenResult.length; i++) {
            hiddenResult[i] = activationS(hiddenResult[i]);
        }

        final var resultMatrix = Ops.multipleTransposedConcurrent(assocLayer, hiddenResultMatrix, maxThreads);

        final var result = ((MatrixF32Interface) resultMatrix).getData();

        for (var i = 0; i < result.length; i++) {
            result[i] = activationA(result[i]);
        }

        return result;
    }

    private float activationS(float x) {
        return x > 0 ? x : 0.0f;
    }

    //TODO: implement softmax
    private float activationA(float x) {
        return 1 / (1 + (float)Math.exp(-x));
    }

    private void generateWeightsS(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = random.nextFloat(-1, 1);
        }
    }

    private void generateWeightsA(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = random.nextFloat(-127, 128);
        }
    }

    public void train(float[] sensorData, float[] target, float speed) {
        final var hiddenResultMatrix = Ops.multipleTransposedConcurrent(sensorLayer, new MatrixF32(1, sensorData.length, sensorData), maxThreads);

        final var hiddenResult = ((MatrixF32Interface) hiddenResultMatrix).getData();

        for (var i = 0; i < hiddenResult.length; i++) {
            hiddenResult[i] = activationS(hiddenResult[i]);
        }

        final var resultMatrix = Ops.multipleTransposedConcurrent(assocLayer, hiddenResultMatrix, maxThreads);

        final var result = ((MatrixF32Interface) resultMatrix).getData();

        for (var i = 0; i < result.length; i++) {
            result[i] = activationA(result[i]);
        }

        for (var i = 0; i < outputLayerSize; i++) {
            float delta = 10 * speed * (target[i] - result[i]);
            for (var j = 0; j < assocLayerSize; j++) {
                assocLayer.getData()[i * assocLayerSize + j] += delta * hiddenResult[j];
            }
        }
    }
}
