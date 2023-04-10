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

    public static void dropout(Random random, float[] result, float k) {
        int[] ints = readIntsFromRandomPool(random, - (int)(result.length * Math.log(1 - k)), 0, result.length);
        for (var i : ints) {
            result[i] = 0.0f;
        }
    }

    public static float dropoutRate(float k) {
        return (1.0f / (1 - k));
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

    public static void deltaCorrection(float alpha, VectorF32 result, float[] target, VectorF32 prevLayerResult, MatrixF32 weights, float l2penalty) {
        var delta = new float[target.length];

        for (var i = 0; i < target.length; i++) {
            delta[i] = alpha * (target[i] - result.getData()[i]);
        }

        Ops.product(prevLayerResult, new VectorF32(delta), weights, 1.0f, 1.0f - l2penalty).getData();
    }
}
