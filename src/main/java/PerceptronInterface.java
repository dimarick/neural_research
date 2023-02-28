import linear.matrix.MatrixF32Interface;

public interface PerceptronInterface {
    float[] eval(float[] sensorData);

    MatrixF32Interface evalLayer1(float[] sensorData);

    float[] trainLayer2(MatrixF32Interface hiddenResultMatrix, float[] target, float speed, boolean useDiff);

    float[] getAssocLayer();

    void setAssocLayer(float[] data);

    float[] train(float[] trainImage, float[] target, float speed);
}
