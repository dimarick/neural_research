
import java.util.Random;

public class SingleLayerPerceptron {
    final private float[][] weights;
    final private int sensorLayerSize;
    final private int outputLayerSize;

    public SingleLayerPerceptron(int sensorLayerSize, int outputLayerSize, long seed) {
        this.sensorLayerSize = sensorLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.weights = new float[outputLayerSize][sensorLayerSize];

        generateWeights(seed);
    }

    public float[] eval(float[] sensorData) {
        var result = new float[outputLayerSize];

        for (var i = 0; i < outputLayerSize; i++) {
            for (var j = 0; j < sensorLayerSize; j++) {
                result[i] += sensorData[j] * weights[i][j];
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

    private void generateWeights(long seed) {
        var random = new Random(seed);
        for (var i = 0; i < outputLayerSize; i++) {
            for (var j = 0; j < sensorLayerSize; j++) {
                this.weights[i][j] = random.nextFloat(0, 1);
            }
        }
    }

    public void train(float[] sensorData, float[] result, float[] target, float speed)
    {
        for (var i = 0; i < outputLayerSize; i++) {
            float delta = speed * (target[i] - result[i]);
            for (var j = 0; j < sensorLayerSize; j++) {
                weights[i][j] += delta * sensorData[j];
            }
        }
    }
}
