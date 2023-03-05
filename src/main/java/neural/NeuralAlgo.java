package neural;

import linear.matrix.MatrixF32;
import linear.matrix.MatrixF32Interface;
import linear.matrix.Ops;

import java.util.Arrays;
import java.util.Random;

public class NeuralAlgo {

    public static void generalizationApply(float l1, MatrixF32Interface weights, float generalizationFactor) {
        var a = weights.getData();
        for (var i = 0; i < a.length; i++) {
            a[i] = a[i] > 0 ? Math.max(0, a[i] - l1 * generalizationFactor) : Math.min(0, a[i] + l1 * generalizationFactor);
        }
    }

    public static float loss(float[] result, float[] target, float threshold) {
        var a = getAnswer(target);

        var loss = 0.0f;

        for (var i = 0; i < result.length; i++) {
            if (a == i) {
                continue;
            }

            loss += Math.abs(Math.max(0, result[i] - result[a] + threshold));
        }

        return loss / result.length;
    }

    public static void dropout(Random random, float[] result, float k) {
        for (var i : random.ints((long)(-result.length * Math.log(1 - k)), 0, result.length).toArray()) {
            result[i] = 0.0f;
        }
    }

    public static float dropoutRate(float k) {
        return (1.0f / (1 - k));
    }

    public static int getAnswer(float[] result) {
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
    public static float generalizeLasso(MatrixF32Interface weights) {
        var result = 0.0f;

        for (float item : weights.getData()) {
            result += Math.abs(item);
        }

        return result / weights.getData().length;
    }

    public static float generalizeRidge(MatrixF32Interface weights) {
        var result = 0.0f;

        for (float item : weights.getData()) {
            result += item * item;
        }

        return result / weights.getData().length;
    }


    public static void normalize(MatrixF32Interface vector) {
        final var data = vector.getData();
        var max = 0.0f;
        var min = 0.0f;

        for (var i = 0; i < data.length; i++) {
            max = Math.max(max, data[i]);
            min = Math.min(min, data[i]);
        }

        for (var i = 0; i < data.length; i++) {
            data[i] = (data[i] - min) / (max - min);
        }
    }

    public static void softmax(MatrixF32Interface vector, float alpha) {
        final var data = vector.getData();
        var sum = 0.0f;

        for (var i = 0; i < data.length; i++) {
            var exp = normalize((float)Math.exp(data[i] * alpha), Float.MAX_VALUE / vector.getData().length);
            sum += exp;
            data[i] = exp;
        }

        sum = normalize(sum, Float.MAX_VALUE);

        if (sum == 0.0f) {
            Arrays.fill(data, 0.0f);

            return;
        }

        for (var i = 0; i < data.length; i++) {
            data[i] /= sum;
        }
    }

    private static float normalize(float x, float max) {
        return Math.max(Math.min(x, max), -max);
    }

    public static float[] softmaxDiff(MatrixF32Interface vector, float alpha) {
        final var data = vector.getData().clone();
        var sum = 0.0f;

        for (var i = 0; i < data.length; i++) {
            var exp = (float)Math.max(Math.min(Math.exp(data[i] * alpha), Float.MAX_VALUE / vector.getData().length / 2), -Float.MAX_VALUE / vector.getData().length / 2);
            sum += exp;
            data[i] = exp;
        }

        sum = Math.max(Math.min(sum, Float.MAX_VALUE), -Float.MAX_VALUE);

        if (sum == 0.0f) {
            Arrays.fill(data, 0.0f);

            return data;
        }

        for (var i = 0; i < data.length; i++) {
            data[i] = (data[i] / sum) * (1 - data[i] / sum);
        }

        return data;
    }

    public static void reLU(MatrixF32Interface vector) {
        final var data = vector.getData();

        for (var i = 0; i < data.length; i++) {
            //noinspection ManualMinMaxCalculation
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
    }

    public static float[] reLUDiff(MatrixF32Interface vector) {
        final var data = vector.getData().clone();

        for (var i = 0; i < data.length; i++) {
            //noinspection ManualMinMaxCalculation
            data[i] = data[i] > 0.0f ? 1.0f : 0.0f;
        }

        return data;
    }

    public interface ActivationDiff {
        float[] eval(MatrixF32Interface result);
    }

    public interface LossFunction {
        float eval(MatrixF32Interface result, float[] target);
    }

    public static void sdg(float alpha, float[] diff, float[] loss, MatrixF32Interface prevLayerResult, MatrixF32Interface weights) {
        var delta = new float[loss.length];

        for (var i = 0; i < loss.length; i++) {
            delta[i] = -alpha * loss[i] * diff[i];
        }

        Ops.multiple(new MatrixF32(delta.length, 1, delta), Ops.transposeVector(prevLayerResult), weights, 1.0f, 1.0f).getData();
    }

    public static void deltaCorrection(float alpha, LossFunction lossFunction, MatrixF32Interface result, float[] target, MatrixF32Interface prevLayerResult, MatrixF32Interface weights) {
        var loss = lossFunction.eval(result, target);

        var delta = new float[target.length];

        for (var i = 0; i < target.length; i++) {
            delta[i] = alpha * loss * (target[i] - result.getData()[i]);
        }

        Ops.multiple(new MatrixF32(target.length, 1, delta), Ops.transposeVector(prevLayerResult), weights, 1.0f, 1.0f).getData();
    }
}
