import linear.matrix.MatrixF32;
import linear.matrix.MatrixF32Interface;
import linear.matrix.Ops;

import java.util.Random;

public class RosenblattMultiLevelPerceptron {
    public static final float ALPHA = 0.1f;
    final private int outputLayerSize;
    final private int assocLayerSize;
    final private MatrixF32 sensorLayer;
    final private MatrixF32 assocLayer;

    public RosenblattMultiLevelPerceptron(int sensorLayerSize, int outputLayerSize, int assocLayerSize, Random random) {
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
        final var hiddenResultMatrix = evalLayer1(sensorData);
        final var resultMatrix = evalLayer2(hiddenResultMatrix);
        Ops.normalize(resultMatrix);
        Ops.softmax(resultMatrix, ALPHA / assocLayerSize * 200);

        return resultMatrix.getData();
    }

    private MatrixF32Interface evalLayer2(MatrixF32Interface hiddenResultMatrix) {
        final var resultMatrix = Ops.multiple(assocLayer, hiddenResultMatrix);
        return resultMatrix;
    }

    public MatrixF32Interface evalLayer1(float[] sensorData) {
        final var hiddenResultMatrix = Ops.multiple(sensorLayer, new MatrixF32(sensorData.length, 1, sensorData));
        Ops.reLU(hiddenResultMatrix);
        Ops.normalize(hiddenResultMatrix);
        return hiddenResultMatrix;
    }

    public float[] trainLayer2(MatrixF32Interface hiddenResultMatrix, float[] target, float speed, boolean useDiff) {
        final var resultMatrix = evalLayer2(hiddenResultMatrix);

        final var result = resultMatrix.getData();
        final var diffMatrix = new MatrixF32(result.length, 1, result.clone());
        final var diff = diffMatrix.getData();

        Ops.softmaxDiff(diffMatrix, ALPHA / assocLayerSize * 400);
        Ops.softmax(resultMatrix, ALPHA / assocLayerSize * 400);

        final var hiddenResult = hiddenResultMatrix.getData();

        float[] data = assocLayer.getData();

        var delta = new float[outputLayerSize];

        for (var i = 0; i < outputLayerSize; i++) {
            float v = target[i] - result[i];
            delta[i] = speed * v * (useDiff ? diff[i] : 1.0f);
        }

//        return Ops.multiple(new MatrixF32(outputLayerSize, 1, delta), Ops.transposeVector(hiddenResultMatrix)).getData();

        for (var i = 0; i < outputLayerSize; i++) {
            for (var j = 0; j < assocLayerSize; j++) {
                data[i * assocLayerSize + j] += delta[i] * hiddenResult[j];
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
