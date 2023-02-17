import java.util.Random;

public class RosenblattPerceptron {
    final private int sensorLayerSize;
    final private int outputLayerSize;
    final private int assocLayerSize;
    final private float[][] sensorLayer;
    final private float[][] assocLayer;

    public RosenblattPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random) {
        this.sensorLayerSize = sensorLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.assocLayerSize = assocLayerSize;

        this.sensorLayer = new float[assocLayerSize][sensorLayerSize];
        this.assocLayer = new float[outputLayerSize][assocLayerSize];

        generateWeightsS(this.sensorLayer, random);
        generateWeightsA(this.assocLayer, random);
    }

    public float[] eval(float[] sensorData) {
        final var hiddenResult = new float[assocLayerSize];

        for (var i = 0; i < assocLayerSize; i++) {
            for (var j = 0; j < sensorLayerSize; j++) {
                hiddenResult[i] += sensorData[j] * sensorLayer[i][j];
            }
        }

        final var result = new float[outputLayerSize];

        for (var i = 0; i < outputLayerSize; i++) {
            for (var j = 0; j < assocLayerSize; j++) {
                result[i] += hiddenResult[j] * assocLayer[i][j];
            }
        }

        for (var i = 0; i < result.length; i++) {
            result[i] = activation(result[i]);
        }

        return result;
    }

    //TODO: implement softmax
    private float activation(float x) {
        return 1 / (1 + (float)Math.exp(-x));
    }

    private void generateWeightsS(float[][] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            for (var j = 0; j < layer[i].length; j++) {
                layer[i][j] = random.nextFloat(random.nextInt(-2550,0), random.nextInt(0,2560));
            }
        }
    }

    private void generateWeightsA(float[][] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            for (var j = 0; j < layer[i].length; j++) {
                layer[i][j] = random.nextFloat(-127, 128);
            }
        }
    }

    public void train(float[] sensorData, float[] target, float speed)
    {
        final var hiddenResult = new float[assocLayerSize];

        for (var i = 0; i < assocLayerSize; i++) {
            for (var j = 0; j < sensorLayerSize; j++) {
                hiddenResult[i] += sensorData[j] * sensorLayer[i][j];
            }
        }

        final var result = new float[outputLayerSize];

        for (var i = 0; i < outputLayerSize; i++) {
            for (var j = 0; j < assocLayerSize; j++) {
                result[i] += hiddenResult[j] * assocLayer[i][j];
            }
        }

        for (var i = 0; i < result.length; i++) {
            result[i] = activation(result[i]);
        }

        for (var i = 0; i < outputLayerSize; i++) {
            float delta = 10 * speed * (target[i] - result[i]);
            for (var j = 0; j < assocLayerSize; j++) {
                assocLayer[i][j] += delta * hiddenResult[j];
            }
        }
    }
}
