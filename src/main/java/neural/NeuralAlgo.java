package neural;

import dev.ludovic.netlib.BLAS;
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

    public static void parallel(BiConsumer<Integer, Integer> task) {
        int cores = Runtime.getRuntime().availableProcessors();
        parallel(task, cores);
    }

    public static void parallel(BiConsumer<Integer, Integer> task, int cores) {
        IntStream.range(0, cores).parallel().forEach(t -> task.accept(t, cores));
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
        float[] data = weights.getData();

        return BLAS.getInstance().sasum(data.length, data, 1) / data.length;
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

        for (float datum : data) {
            max = Math.max(max, datum);
            min = Math.min(min, datum);
        }

        for (var i = 0; i < data.length; i++) {
            data[i] = (data[i] - min) / (max - min);
        }
    }


    public interface LossFunction {
        float eval(VectorF32 result, float[] target);
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
