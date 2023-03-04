import linear.matrix.MatrixF32;
import linear.matrix.MatrixF32Interface;
import linear.matrix.Ops;

import java.util.*;

/**
 * Реализации перцептрона Розенблатта. От классического отличается наличием дропаута, генерализации, небинарного входа.
 * Все это позволило получить характеристики выше чем в "Rosenblatt Perceptrons for Handwritten Digit Recognition" вплоть до размера слоя 32000.
 * Дальнейшее совершенствование требует расширения обучаещего набора путем аугментации или другими методами за рамками этой работы.
 * При A = 250 (0,4 * S) показывает результат лучше по сравнению с наивной реализацией, что показывает корректность его реализации
 * Увеличение размера ассоциативного слоя позволяет повышать точность, что недостижимо в наивной реализации
 * При этом он существенно медленнее работает из-за умножения матриц размерностями S и S*A
 * Качество распознавания на 7000 А-нейронах составило 380-420 ошибок, что превосходит реализациям
 * из публикации "Rosenblatt Perceptrons for Handwritten Digit Recognition" (предположительно из-за небинарного входного слоя),
 * однако далее качество не растет (или растет крайне медленно). Здесь, в отличие от их реализации связи S-A полностью
 * случайны. Опытным путем установлено что выбор разного seed для их генерации влияет на качество результата порядка 5-10%,
 * также очевидно что оптимальный подбор весов повысит качество еще немного, что видно из исследования. На этом возможности однослойных перцептронов считаю исчерпанными
 * Эффективность до 97,89%
 */
public class RosenblattPerceptron {
    public static final float ALPHA = 1.0f;
    public static final float GENERALIZATION_FACTOR = 1e-6f;
    public static final float LOSS_THRESHOLD = 0.7f;
    final private int outputLayerSize;
    final private Random random;
    final private MatrixF32 sensorLayer;
    final private MatrixF32 assocLayer;

    public RosenblattPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random) {
        this.outputLayerSize = outputLayerSize;
        this.random = new Random(random.nextLong());

        this.sensorLayer = new MatrixF32(assocLayerSize, sensorLayerSize);
        this.assocLayer = new MatrixF32(outputLayerSize, assocLayerSize);

        generateWeights(this.sensorLayer.getData(), random);
        generateWeights(this.assocLayer.getData(), random);
    }

    private void generateWeights(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = (float)random.nextGaussian(0.0f, 1f);
        }
    }

    public float[] eval(float[] sensorData) {
        final var hiddenResultMatrix = evalLayer1(sensorData);
        final var resultMatrix = evalLayer2(hiddenResultMatrix);
        Ops.normalize(resultMatrix);
        Ops.softmax(resultMatrix, ALPHA);

        return resultMatrix.getData();
    }

    private MatrixF32Interface evalLayer2(MatrixF32Interface hiddenResultMatrix) {
        return Ops.multiple(assocLayer, hiddenResultMatrix);
    }

    public MatrixF32Interface evalLayer1(float[] sensorData) {
        final var hiddenResultMatrix = Ops.multiple(sensorLayer, new MatrixF32(sensorData.length, 1, sensorData));
        Ops.reLU(hiddenResultMatrix);
        Ops.normalize(hiddenResultMatrix);

        return hiddenResultMatrix;
    }

    public float[] trainLayer2(MatrixF32Interface hiddenResultMatrix, float[] target, float speed, float dropoutFactor) {
        hiddenResultMatrix = new MatrixF32(hiddenResultMatrix.getData().length, 1, hiddenResultMatrix.getData().clone());

        if (dropoutFactor > 0) {
            Ops.dropout(random, hiddenResultMatrix.getData(), dropoutFactor);
        }

        final var resultMatrix = evalLayer2(hiddenResultMatrix);

        final var result = resultMatrix.getData();

        Ops.softmax(resultMatrix, ALPHA);

        var loss = Ops.loss(result, target, LOSS_THRESHOLD);

        var delta = new float[outputLayerSize];

        float alpha = speed * loss * Ops.dropoutRate(dropoutFactor);

        for (var i = 0; i < outputLayerSize; i++) {
            delta[i] = alpha * (target[i] - result[i]);
        }

        if (random.nextFloat(0.0f, 1.0f) > 0.95f) {
            float l1 = Ops.generalizeLasso(assocLayer);
//            float l1 = generalizeRidge(assocLayer);
//
            Ops.generalizationApply(l1, assocLayer, GENERALIZATION_FACTOR);
        }

        Ops.multiple(new MatrixF32(outputLayerSize, 1, delta), Ops.transposeVector(hiddenResultMatrix), assocLayer, 1.0f, 1.0f).getData();

        return result;
    }

    public float[] train(float[] sensorData, float[] target, float speed, float dropoutFactor) {
        final var hiddenResultMatrix = evalLayer1(sensorData);
        return trainLayer2(hiddenResultMatrix, target, speed, dropoutFactor);
    }
}
