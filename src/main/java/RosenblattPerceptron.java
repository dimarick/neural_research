import linear.matrix.MatrixF32;
import linear.matrix.Ops;

import java.util.Random;

/**
 * Классическая реализации перцептрона Розенблатта
 * При A = 250 (0,4 * S) показывает результат лучше по сравнению с наивной реализацией, что показывает корректность его реализации
 * Увеличение размера ассоциативного слоя позволяет повышать точность, что недостижимо в наивной реализации
 * При этом он существенно медленнее работает из-за умножения матриц размерностями S и S*A
 * Качество распознавания на 7000 А-нейронах составило 380-420 ошибок, что превосходит реализациям
 * из публикации "Rosenblatt Perceptrons for Handwritten Digit Recognition" (предположительно из-за небинарного входного слоя),
 * однако далее качество не растет (или растет крайне медленно). Здесь, в отличие от их реализации связи S-A полностью
 * случайны. Опытным путем установлено что выбор разного seed для их генерации влияет на качество результата порядка 5-10%,
 * также очевидно что оптимальный подбор весов повысит качество еще немного, что видно из исследования. На этом возможности однослойных перцептронов считаю исчерпанными
 */
public class RosenblattPerceptron {
    final private int outputLayerSize;
    final private int assocLayerSize;
    final private MatrixF32 sensorLayer;
    final private MatrixF32 assocLayer;

    public RosenblattPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random, int maxThreads) {
        this.outputLayerSize = outputLayerSize;
        this.assocLayerSize = assocLayerSize;

        this.sensorLayer = new MatrixF32(assocLayerSize, sensorLayerSize);
        this.assocLayer = new MatrixF32(outputLayerSize, assocLayerSize);

        generateWeightsS(this.sensorLayer.getData(), random);
        generateWeightsA(this.assocLayer.getData(), random);
    }

    private void generateWeightsS(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = random.nextFloat(-1, 1);
        }
    }

    private void generateWeightsA(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = random.nextFloat(-1, 1);
        }
    }

    public float[] eval(float[] sensorData) {
        final var hiddenResultMatrix = Ops.multiple(sensorLayer, new MatrixF32(sensorData.length, 1, sensorData));
        Ops.reLU(hiddenResultMatrix);
        Ops.normalize(hiddenResultMatrix);
        final var resultMatrix = Ops.multiple(assocLayer, hiddenResultMatrix);
        Ops.logisticFn(resultMatrix, 1.0f);

        return resultMatrix.getData();
    }

    public float[] train(float[] sensorData, float[] target, float speed) {
        final var hiddenResultMatrix = Ops.multiple(sensorLayer, new MatrixF32(sensorData.length, 1, sensorData));
        Ops.reLU(hiddenResultMatrix);
        Ops.normalize(hiddenResultMatrix);
        final var resultMatrix = Ops.multiple(assocLayer, hiddenResultMatrix);
        Ops.logisticFn(resultMatrix, 1.0f);

        final var result = resultMatrix.getData();
        final var hiddenResult = hiddenResultMatrix.getData();

        for (var i = 0; i < outputLayerSize; i++) {
            float delta = speed * (target[i] - result[i]);
            for (var j = 0; j < assocLayerSize; j++) {
                assocLayer.getData()[i * assocLayerSize + j] += delta * hiddenResult[j];
            }
        }

        return result;
    }

    public float[] getAssocLayer() {
        return assocLayer.getData().clone();
    }

    public void setAssocLayer(float[] data) {
        assocLayer.setData(data);
    }
}
