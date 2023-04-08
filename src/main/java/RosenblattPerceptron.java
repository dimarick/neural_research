import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;
import neural.Activation;
import neural.NeuralAlgo;

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
    final private Random random;
    final private MatrixF32 sensorLayer;
    final private MatrixF32 assocLayer;

    public RosenblattPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random) {
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
        NeuralAlgo.normalize(resultMatrix);
        new Activation.Softmax(ALPHA).apply(resultMatrix);

        return resultMatrix.getData();
    }

    private VectorF32 evalLayer2(VectorF32 hiddenResultMatrix) {
        return Ops.product(assocLayer, hiddenResultMatrix);
    }

    public VectorF32 evalLayer1(float[] sensorData) {
        final var hiddenResultMatrix = Ops.product(sensorLayer, new VectorF32(sensorData));
        new Activation.ReLU().apply(hiddenResultMatrix);
        NeuralAlgo.normalize(hiddenResultMatrix);

        return hiddenResultMatrix;
    }

    public float[] trainLayer2(VectorF32 hiddenResultMatrix, float[] target, float speed, float dropoutFactor) {
        hiddenResultMatrix = new VectorF32(hiddenResultMatrix.getData().clone());

        if (dropoutFactor > 0) {
            NeuralAlgo.dropout(random, hiddenResultMatrix.getData(), dropoutFactor);
        }

        final var resultMatrix = evalLayer2(hiddenResultMatrix);

        final var result = resultMatrix.getData();

        new Activation.Softmax(ALPHA).apply(resultMatrix);

        NeuralAlgo.deltaCorrection(
                speed * NeuralAlgo.dropoutRate(dropoutFactor),
                resultMatrix,
                target,
                hiddenResultMatrix,
                assocLayer
        );

        if (random.nextFloat(0.0f, 1.0f) > 0.95f) {
            float l1 = NeuralAlgo.generalizeLasso(assocLayer);
//            float l1 = generalizeRidge(assocLayer);
//
            NeuralAlgo.generalizationApply(l1, assocLayer, GENERALIZATION_FACTOR);
        }

        return result;
    }

    public float[] train(float[] sensorData, float[] target, float speed, float dropoutFactor) {
        final var hiddenResultMatrix = evalLayer1(sensorData);
        return trainLayer2(hiddenResultMatrix, target, speed, dropoutFactor);
    }
}
