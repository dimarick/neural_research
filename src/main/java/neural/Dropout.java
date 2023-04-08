package neural;

import linear.MatrixF32;
import linear.VectorF32;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;

public class Dropout {
    public interface Interface {
        int[] init(int size);
        void apply(VectorF32 result, int[] indexes);
        void apply(MatrixF32 result, int[] indexes);
        float getRate();
    }

    public static class Zero implements Interface {
        final protected Random random;
        public float k;

        public Zero(Random random, float k) {
            this.random = random;
            this.k = k;
        }

        @Override
        public int[] init(int size) {
            return getInts(size);
        }

        public void apply(VectorF32 result, int[] indexes) {
            float[] resultData = result.getData();

            for (var i :indexes) {
                resultData[i] = 0.0f;
            }
        }

        public void apply(MatrixF32 result, int[] indexes) {
            float[] resultData = result.getData();

            for (var i :indexes) {
                resultData[i] = 0.0f;
            }
        }

        public float getRate() {
            return (1.0f / (1 - k));
        }

        protected int[] getInts(int size) {
            if (size == 0) {
                return new int[0];
            }

            return readIntsFromRandomPool(random, -(int) (size * Math.log(1 - k)), 0, size);
        }
    }

    public static class Rng extends Zero {
        public Rng(Random random, float k) {
            super(random, k);
        }

        public void apply(VectorF32 result, int[] indexes) {
            var values = readIntsFromRandomPool(random, indexes.length);
            float[] resultData = result.getData();

            var scale = 1.0f / ((long)Integer.MAX_VALUE - (long)Integer.MIN_VALUE);

            for (var i = 0; i < indexes.length; i++) {
                var index = indexes[i];
                if (index > resultData.length) {
                    continue;
                }
                if (index > values.length) {
                    continue;
                }
                resultData[index] = values[i] * scale + 0.5f;
            }
        }
    }

    final private static int[] randomPool = new int[1048576 * 1024];
    private static int randomPoolCursor = -1;

    private static void parallel(BiConsumer<Integer, Integer> task) {
        int cores = Runtime.getRuntime().availableProcessors();
        IntStream.rangeClosed(0, cores).parallel().forEach(t -> task.accept(t, cores));
    }

    private static int[] readIntsFromRandomPool(Random random, int n, int min, int max) {
        enshurePoolFullfilled(random, n);

        var scale = (double)((long)Integer.MAX_VALUE - (long)Integer.MIN_VALUE);
        var result = new int[n];
        int range = max - min;

        for (var i = 0; i < n; i++) {
            var d = randomPool[i + randomPoolCursor - n];
            result[i] = (int)((d / scale + 0.5) * range + min);
        }

        return result;
    }

    private static int[] readIntsFromRandomPool(Random random, int n) {
        enshurePoolFullfilled(random, n);

        return Arrays.copyOfRange(randomPool, randomPoolCursor - n, randomPoolCursor);
    }

    private static void enshurePoolFullfilled(Random random, int n) {
        if (randomPoolCursor == -1) {
            fulfillPool(random);

            randomPoolCursor = 0;
        }
        if (randomPoolCursor + n >= randomPool.length) {

            randomPoolCursor = 0;
        }

        randomPoolCursor += n;
    }

    private static void fulfillPool(Random random) {
        parallel((t, cores) -> {
            var r = new Random(random.nextInt());
            var batch = Math.ceil((double) randomPool.length / cores);
            var start = (int)(t * batch);
            var end = (int)Math.min(randomPool.length, start + batch);

            for (var i = start; i < end; i++) {
                randomPool[i] = r.nextInt();
            }
        });
    }
}
