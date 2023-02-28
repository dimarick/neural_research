import linear.matrix.MatrixF32;
import linear.matrix.MatrixF32Interface;
import linear.matrix.Ops;

import java.util.ArrayList;
import java.util.Collections;
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
 * Эффективность до 97,8%
 */
public class RosenblattPerceptron implements PerceptronInterface {
    public static final float ALPHA = 1.0f;
    public static final float GENERALIZATION_FACTOR = 1.0f;
    public static final float LOSS_THRESHOLD = 1.0f;
    public static final float DROPOUT_FACTOR = 0.4f;
    final private int outputLayerSize;
    final private int assocLayerSize;
    final private MatrixF32 sensorLayer;
    final private MatrixF32 assocLayer;
    final private ArrayList<Integer> dropList;

    public RosenblattPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random) {
        this.outputLayerSize = outputLayerSize;
        this.assocLayerSize = assocLayerSize;

        this.sensorLayer = new MatrixF32(assocLayerSize, sensorLayerSize);
        this.assocLayer = new MatrixF32(outputLayerSize, assocLayerSize);

        generateWeightsS(this.sensorLayer.getData(), random);
        generateWeightsA(this.assocLayer.getData(), random);

        dropList = new ArrayList<>(assocLayerSize);
        for (var i = 0; i < assocLayerSize; i++) {
            dropList.add(i);
        }
    }

    private void generateWeightsS(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = (float)random.nextGaussian(0.0f, 1f);
        }
    }

    private void generateWeightsA(float[] layer, Random random) {
        for (var i = 0; i < layer.length; i++) {
            layer[i] = random.nextFloat(-1, 1);
        }
    }

    @Override
    public float[] eval(float[] sensorData) {
        final var hiddenResultMatrix = evalLayer1(sensorData);
        final var resultMatrix = evalLayer2(hiddenResultMatrix);
        Ops.normalize(resultMatrix);
        Ops.softmax(resultMatrix, ALPHA / assocLayerSize);

        return resultMatrix.getData();
    }

    private MatrixF32Interface evalLayer2(MatrixF32Interface hiddenResultMatrix) {
        return Ops.multiple(assocLayer, hiddenResultMatrix);
    }

    @Override
    public MatrixF32Interface evalLayer1(float[] sensorData) {
        final var hiddenResultMatrix = Ops.multiple(sensorLayer, new MatrixF32(sensorData.length, 1, sensorData));
        Ops.reLU(hiddenResultMatrix);
        Ops.normalize(hiddenResultMatrix);

        return hiddenResultMatrix;
    }

    @Override
    public float[] trainLayer2(MatrixF32Interface hiddenResultMatrix, float[] target, float speed, boolean useDiff) {
        hiddenResultMatrix = new MatrixF32(hiddenResultMatrix.getData().length, 1, hiddenResultMatrix.getData().clone());
        dropout(hiddenResultMatrix.getData(), DROPOUT_FACTOR);
        final var resultMatrix = evalLayer2(hiddenResultMatrix);

        final var result = resultMatrix.getData();

        var diffMatrix = new MatrixF32(resultMatrix.getData().length, 1, resultMatrix.getData().clone());
        Ops.softmaxDiff(diffMatrix, ALPHA / assocLayerSize / 400);
        Ops.softmax(resultMatrix, ALPHA / assocLayerSize);

        var loss = loss(result, target, LOSS_THRESHOLD);

        var delta = new float[outputLayerSize];

        for (var i = 0; i < outputLayerSize; i++) {
            float v = target[i] - result[i];
            delta[i] =  v * diffMatrix.getData()[i];
        }

        if (new Random().nextFloat(0.0f, 1.0f) > 0.9f) {
        float l1 = generalizeLasso();
//            float l1 = generalizeRidge();

            generalizationApply(l1);
        }

        Ops.multiple(new MatrixF32(outputLayerSize, 1, delta), Ops.transposeVector(hiddenResultMatrix), assocLayer, speed * loss * assocLayerSize, 1.0f).getData();

        return result;
    }


    public float[] train(float[] sensorData, float[] target, float speed) {
        final var hiddenResultMatrix = evalLayer1(sensorData);
        return trainLayer2(hiddenResultMatrix, target, speed, false);
    }

    private void generalizationApply(float l1) {
        var a = assocLayer.getData();
        for (var i = 0; i < assocLayerSize; i++) {
            a[i] = a[i] > 0 ? Math.max(0, a[i] - l1 * GENERALIZATION_FACTOR * assocLayerSize) : Math.min(0, a[i] + l1 * GENERALIZATION_FACTOR * assocLayerSize);
        }
    }

    private float loss(float[] result, float[] target, float threshold) {
        var a = getAnswer(target);

        var loss = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (a == i) {
                continue;
            }

            loss = Math.abs(Math.max(0, result[i] - result[a] + threshold));
        }

        return loss / result.length;
    }

    private void dropout(float[] result, float k) {
        Collections.shuffle(dropList);

        for (var i : dropList.subList(0, (int)(result.length * k))) {
            result[i] = 0.0f;
        }

        Ops.multiple(new MatrixF32(result.length, 1, result), new MatrixF32(1, 1, new float[]{1.0f / (1 - k)}));
    }

    private static int getAnswer(float[] result) {
        var a = 0;
        var max = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (result[i] > max) {
                max = result[i];
                a = i;
            }
        }

        return a;
    }
    private float generalizeLasso() {
        float[] data = assocLayer.getData();
        var result = 0.0f;

        for (var i = 0; i < data.length; i++) {
            result += Math.abs(data[i]);
        }

        return result / data.length;
    }

    private float generalizeRidge() {
        float[] data = assocLayer.getData();
        var result = 0.0f;

        for (var i = 0; i < data.length; i++) {
            result += data[i] * data[i];
        }

        return result / data.length;
    }

    public float[] getAssocLayer() {
        return assocLayer.getData().clone();
    }

    public void setAssocLayer(float[] data) {
        assocLayer.setData(data);
    }
}
