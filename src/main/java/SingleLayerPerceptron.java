
import java.util.Random;

public class SingleLayerPerceptron {
    final private float[][] weights;
    final private int sensorLayerSize;
    final private int outputLayerSize;

    public SingleLayerPerceptron(int sensorLayerSize, int outputLayerSize, long seed) {
        this.sensorLayerSize = sensorLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.weights = new float[sensorLayerSize][outputLayerSize];

        generateWeights(seed);
    }

    public float[] eval(float[] sensorData) {
        var result = new float[outputLayerSize];
        
        for (var i = 0; i < sensorLayerSize; i++) {
            for (var j = 0; j < outputLayerSize; j++) {
                result[j] += sensorData[i] * weights[i][j];
            }
        }

        for (var i = 0; i < result.length; i++) {
            result[i] = activation(result[i]);
        }

        return result;
    }

    private float activation(float x) {
        return 1 / (1 + (float)Math.exp(-x / 400));
    }

    private void generateWeights(long seed) {
        var random = new Random(seed);
        for (var i = 0; i < sensorLayerSize; i++) {
            for (var j = 0; j < outputLayerSize; j++) {
                this.weights[i][j] = random.nextFloat(0, 1);
            }
        }
    }

    public void train(float[] sensorData, float[] result, float[] target, double speed)
    {
        for (var i = 0; i < sensorLayerSize; i++) {
            for (var j = 0; j < outputLayerSize; j++) {
                weights[i][j] -= speed * result[j] * (result[j] - target[j]) * sensorData[i];
            }
        }
    }
}
