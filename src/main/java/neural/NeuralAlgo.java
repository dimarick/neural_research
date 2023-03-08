package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

public class NeuralAlgo {

    final private static int[] randomPool = new int[1048576];
    private static int randomPoolCursor = randomPool.length;

    private static void parallel(BiConsumer<Integer, Integer> task) {
        int cores = Runtime.getRuntime().availableProcessors();
        IntStream.rangeClosed(0, cores).parallel().forEach(t -> {
            task.accept(t, cores);
        });
    }

    private static int[] readIntsFromRandomPool(Random random, int n, int min, int max) {
        if (randomPoolCursor + n >= randomPool.length) {
            parallel((t, cores) -> {
                var r = new Random(random.nextInt());
                var batch = Math.ceil((double) randomPool.length / cores);
                var start = (int)(t * batch);
                var end = (int)Math.min(randomPool.length, start + batch);

                for (var i = start; i < end; i++) {
                    randomPool[i] = r.nextInt();
                }
            });

            randomPoolCursor = 0;
        }

        randomPoolCursor += n;
        var scale = (double)((long)Integer.MAX_VALUE - (long)Integer.MIN_VALUE);

        return Arrays.stream(Arrays
                .copyOfRange(randomPool, randomPoolCursor - n, randomPoolCursor))
                .map(d -> (int)((d / scale + 0.5) * (max - min) + min)).toArray();
    }

    public static void generalizationApply(float l1, MatrixF32 weights, float generalizationFactor) {
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
        int[] ints = readIntsFromRandomPool(random, - (int)(result.length * Math.log(1 - k)), 0, result.length);
        for (var i : ints) {
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
    public static float generalizeLasso(MatrixF32 weights) {
        var result = 0.0f;

        for (float item : weights.getData()) {
            result += Math.abs(item);
        }

        return result / weights.getData().length;
    }

    public static float generalizeRidge(MatrixF32 weights) {
        var result = 0.0f;

        for (float item : weights.getData()) {
            result += item * item;
        }

        return result / weights.getData().length;
    }

    public static void normalize(VectorF32 vector) {
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

    public static void softmax(VectorF32 vector, float alpha) {
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

    public static float[] softmaxDiff(VectorF32 vector, float alpha) {
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

    public static void reLU(VectorF32 vector) {
        final var data = vector.getData();

        for (var i = 0; i < data.length; i++) {
            //noinspection ManualMinMaxCalculation
            data[i] = data[i] > 0.0f ? data[i] : 0.0f;
        }
    }

    public static float[] reLUDiff(VectorF32 vector) {
        final var data = vector.getData().clone();

        for (var i = 0; i < data.length; i++) {
            data[i] = data[i] > 0.0f ? 1.0f : 0.0f;
        }

        return data;
    }

    public interface ActivationDiff {
        float[] eval(MatrixF32 result);
    }

    public interface LossFunction {
        float eval(VectorF32 result, float[] target);
    }

    public static void sdg(float alpha, float[] diff, float[] loss, VectorF32 prevLayerResult, MatrixF32 weights) {
        var delta = new float[loss.length];

        for (var i = 0; i < loss.length; i++) {
            delta[i] = -alpha * loss[i] * diff[i];
        }

        Ops.multiple(new VectorF32(delta), prevLayerResult, weights, 1.0f, 1.0f).getData();
    }

    public static void deltaCorrection(float alpha, LossFunction lossFunction, VectorF32 result, float[] target, VectorF32 prevLayerResult, MatrixF32 weights) {
        var loss = lossFunction.eval(result, target);

        var delta = new float[target.length];

        for (var i = 0; i < target.length; i++) {
            delta[i] = alpha * loss * (target[i] - result.getData()[i]);
        }

        Ops.multiple(new VectorF32(delta), prevLayerResult, weights, 1.0f, 1.0f).getData();
    }
}
