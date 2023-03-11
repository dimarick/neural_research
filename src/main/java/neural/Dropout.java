package neural;

import linear.VectorF32;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

public class Dropout {
    public interface Interface {
        void apply(VectorF32 result);
        float getRate(VectorF32 result);
    }

    public static class Zero implements Interface {
        final protected Random random;
        final protected float k;

        public Zero(Random random, float k) {
            this.random = random;
            this.k = k;
        }

        public void apply(VectorF32 result) {
            int[] ints = getInts(result);
            float[] resultData = result.getData();

            for (var i :ints) {
                resultData[i] = 0.0f;
            }
        }

        public float getRate(VectorF32 result) {
            return (1.0f / (1 - k));
        }

        protected int[] getInts(VectorF32 result) {
            if (result.getSize() == 0) {
                return new int[0];
            }

            return readIntsFromRandomPool(random, -(int) (result.getSize() * Math.log(1 - k)), 0, result.getSize());
        }
    }

    public static class Rng extends Zero {
        public Rng(Random random, float k) {
            super(random, k);
        }

        public void apply(VectorF32 result) {
            int[] ints = getInts(result);
            float[] resultData = result.getData();

            for (var i :ints) {
                resultData[i] = random.nextFloat(0.0f, 1.0f);
            }
        }
    }

    final private static int[] randomPool = new int[1048576];
    private static int randomPoolCursor = randomPool.length;

    private static void parallel(BiConsumer<Integer, Integer> task) {
        int cores = Runtime.getRuntime().availableProcessors();
        IntStream.rangeClosed(0, cores).parallel().forEach(t -> task.accept(t, cores));
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
}
