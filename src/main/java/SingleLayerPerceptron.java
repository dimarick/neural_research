
import linear.matrix.MatrixF32;
import linear.matrix.Ops;

import java.util.Random;

/**
 * Наивная реализация перцептрона, упрощеннная версия перцептрона Розенблатта,
 * в которой объединены S и А слои в один S слой
 * Существенно выигрывает в производительности (раз в 20 для изображений 28*28, если S = A)
 * Предел качества распознавания - 8 % ошибок
 */
public class SingleLayerPerceptron {
    final private MatrixF32 weights;
    final private int sensorLayerSize;
    final private int outputLayerSize;

    public SingleLayerPerceptron(int sensorLayerSize, int outputLayerSize, long seed) {
        this.sensorLayerSize = sensorLayerSize;
        this.outputLayerSize = outputLayerSize;
        this.weights = new MatrixF32(outputLayerSize, sensorLayerSize);

        generateWeights(seed);
    }

    public float[] eval(float[] sensorData) {
        var sensor = new MatrixF32(1, sensorData.length, sensorData);
        var result = Ops.multipleTransposed(this.weights, sensor).getData();

        for (var i = 0; i < result.length; i++) {
            result[i] = activation(result[i]);
        }

        return result;
    }

    private float activation(float x) {
        return 1 / (1 + (float)Math.exp(-x));
    }

    private void generateWeights(long seed) {
        var random = new Random(seed);
        var data = this.weights.getData();
        for (var i = 0; i < data.length; i++) {
            data[i] = random.nextFloat(0, 1);
        }
    }

    public void train(float[] sensorData, float[] target, float speed) {
        var sensor = new MatrixF32(1, sensorData.length, sensorData);
        var result = Ops.multipleTransposed(this.weights, sensor).getData();

        for (var i = 0; i < result.length; i++) {
            result[i] = activation(result[i]);
        }

        var weights = this.weights.getData();

        for (var i = 0; i < outputLayerSize; i++) {
            float delta = speed * (target[i] - result[i]);
            int i1 = i * this.weights.columns;
            for (var j = 0; j < sensorLayerSize; j++) {
                weights[i1 + j] += delta * sensorData[j];
            }
        }
    }
}
