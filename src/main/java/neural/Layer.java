package neural;

import linear.MatrixF32;

import java.util.Random;

public final class Layer {
    public final RumelhartPerceptron parent;
    public final int size;
    public final MatrixF32 weights;
    public Activation.Interface activation;
    public Loss.Interface loss;
    public Regularization.Interface regularization;
    public Dropout.Interface dropout;

    public Layer(RumelhartPerceptron parent, int size, MatrixF32 weights) {
        this.parent = parent;
        this.size = size;
        this.weights = weights;
        this
                .set(new Activation.Softmax())
                .set(new Regularization.Lasso())
                .set(new Dropout.Zero(new Random(), 0));
    }

    public RumelhartPerceptron parent() {
        return parent;
    }

    public Layer set(Activation.Interface activation) {
        this.activation = activation;
        this.set(activation.suggestLoss());

        return this;
    }

    public Layer set(Loss.Interface loss) {
        this.loss = loss;

        return this;
    }

    public Layer set(Regularization.Interface regularization) {
        this.regularization = regularization;

        return this;
    }

    public Layer set(Dropout.Interface dropout) {
        this.dropout = dropout;

        return this;
    }
}
