import com.google.common.primitives.Floats;
import linear.matrix.MatrixF32;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class RosenblattTest5 {

    private static final int EPOCHS = 100;
    private static final float SPEED_SCALE_UP = 1.15f;
    private static final float INITIAL_SPEED = 0.3f;

    public static void main(String[] args) throws RuntimeException {
        try (
                var testImagesFile = new FileInputStream("src/main/resources/t10k-images-idx3-ubyte.gz");
                var testLabelsFile = new FileInputStream("src/main/resources/t10k-labels-idx1-ubyte.gz");
                var trainImagesFile = new FileInputStream("src/main/resources/train-images-idx3-ubyte.gz");
                var trainLabelsFile = new FileInputStream("src/main/resources/train-labels-idx1-ubyte.gz")
        ) {
            var start = System.currentTimeMillis();

            var testImages = getImages(testImagesFile);
            var testLabels = getLabels(testLabelsFile);
            var trainImages = getImages(trainImagesFile);
            var trainLabels = getLabels(trainLabelsFile);

            var loaded = System.currentTimeMillis() - start;

            System.out.println("Files loaded " + loaded + " ms");

            var result = trainImages.length;

            for (var i = 0; i < 30; i++) {
                var a = 125 * Math.pow(2, i);
                var speed = INITIAL_SPEED;

                System.out.println("Starting test with speed " + speed + "(" + a + ")");
                var p = new RosenblattPerceptron(28 * 28, 10, (int) (a), new SecureRandom(new byte[]{3}));
                result = train(testImages, testLabels, trainImages, trainLabels, speed, 0.01f, p);

                var testStart = System.currentTimeMillis();

                var fail = test(testImages, testLabels, p);

                var trainRate = ((float)result / trainImages.length) * 100;
                var testRate = (fail / testImages.length) * 100;
                System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + trainRate + "% " + ". Test Error rate is: " + testRate + "%");
            }

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int train(float[][] testImages, byte[] testLabels, float[][] trainImages, byte[] trainLabels, float speed, float dropout, RosenblattPerceptron p) {
        var order = new LinkedList<Integer>();

        for (var i = 0; i < trainImages.length; i++) {
            order.add(i);
        }

        var layer1 = new MatrixF32[trainImages.length];

        for (var i = 0; i < trainImages.length; i++) {
            layer1[i] = p.evalLayer1(trainImages[i]);
        }

        var fail = 0;
        var prevFail = trainImages.length;
        var failRate = 0.1f;
        var bestEffectiveSpeed = 0.0f;

        var effectiveSpeedQueue = new ArrayList<>(Floats.asList(new float[8]));
        var speedScale = SPEED_SCALE_UP;

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();

            // Перемешивание образцов ускоряет сходимость сети
            Collections.shuffle(order);

            for (var i : order) {
                byte label = trainLabels[i];
                var target = createTargetForLabel(label);
                var r = p.trainLayer2(layer1[i], target, speed, dropout);
                if (getAnswer(r) != label) {
                    fail++;
                }
            }

            failRate = ((float) fail / trainImages.length);

            var testFail = test(testImages, testLabels, p);

            var epochTime = System.currentTimeMillis() - epochStart;
            float testRate = (testFail / testImages.length);

            var effectiveSpeed = (float)prevFail / fail - 1;

            float bias = testRate * 100 - failRate * 100;

            var avgSpeed = (float)effectiveSpeedQueue.stream().mapToDouble(i -> (double) i).average().orElse(0.0f);

            if (avgSpeed > bestEffectiveSpeed && epoch > 4) {
                bestEffectiveSpeed = avgSpeed;
            }

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed + ". Test error rate is: " + testRate * 100 + "%. bias: " + bias + "%. dropout: " + dropout * 100 + "%. Effective speed: " + avgSpeed * 100);

            // Идея автоматического выбора скорости базируется на двух экспериментально установленных фактах:
            // Эффективность обучения при выборе скорости больше оптимума снижается более резко, чем при
            // Оптимальная скорость обучения в процессе смещается не слишком быстро и предсказуемо
            // Поэтому мы начиная с заведома малой скорости ее поднимаем пока скорость не начнет устойчиво падать
            // После этого откатываем скорость на 4 шага назад и уменьшаем ускорение.

            effectiveSpeedQueue.add(effectiveSpeed);
            effectiveSpeedQueue.remove(0);

            var speed0 = effectiveSpeedQueue.subList(2, effectiveSpeedQueue.size() - 3).stream().mapToDouble(i -> (double)i).average().orElse(0.0);
            var speed1 = effectiveSpeedQueue.subList(1, effectiveSpeedQueue.size() - 3).stream().mapToDouble(i -> (double)i).average().orElse(0.0);
            var speed2 = effectiveSpeedQueue.subList(0, effectiveSpeedQueue.size() - 3).stream().mapToDouble(i -> (double)i).average().orElse(0.0);

            if (bias > 1.0) {
                dropout = 1 - (1 - dropout) * 0.99f;
            } else {
                if (speed1 > speed0 && speed1 > speed2) {
                    speed = speed / (float) Math.pow(speedScale, 5);
                    speedScale = 1 + 0.9f * (speedScale - 1);
                } else if (speed0 < 0 && speed1 < 0 && speed2 < 0) {
                    speed = speed / (float) Math.pow(speedScale, 3);
                } else {
                    speed *= speedScale;
                }
            }

            prevFail = fail;

            if (fail == 0) {
                break;
            }
        }

//        var testStart = System.currentTimeMillis();

//        var testFail = test(testImages, testLabels, p);

//        var testTime = System.currentTimeMillis() - testStart;

//        System.out.println("test done. " + testTime + " ms. Error rate is: " + (testFail / testImages.length) * 100 + "%");

        return fail;
    }

    private static float test(float[][] testImages, byte[] testLabels, RosenblattPerceptron p) {

        var fail = 0.0f;

        for (var i = 0; i < testImages.length; i++) {
            byte label = testLabels[i];
            var result = p.eval(testImages[i]);

            int answer = getAnswer(result);

            if (answer != label) {
//                output.println("Wrong!!!  i = " + i + ";" + getDistanceEst(result, target) + ", answer is " + answer + "(" + result[answer] + ") != " + label + "    " + Arrays.toString(result));
                fail++;
            }
        }

        return fail;
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

    private static float[] createTargetForLabel(byte label) {
        var result = new float[10];

        result[label] = 1.0f;

        return result;
    }

    private static float[][] getImages(FileInputStream testFile) throws IOException {
        final var testImages = new GZIPInputStream(testFile);
        final var testImagesData = new DataInputStream(testImages);
        final var magic = testImagesData.readInt();

        if (magic != 2051) {
            throw new IOException("Magick is invalid: " + magic);
        }

        final var count = testImagesData.readInt();
        final var rows = testImagesData.readInt();
        final var columns = testImagesData.readInt();
        final var images = new float[count][rows * columns];

        int image = 0;

        try {
            while (image < count) {
                byte[] bytes = testImagesData.readNBytes(rows * columns);

                for (var i = 0; i < bytes.length; i++) {
                    images[image][i] = normalize(bytes[i]);
                }

                image++;
            }
        } catch (EOFException e) {
            System.out.println("EOF: " + e.getMessage());
        }

        return images;
    }

    private static float normalize(byte b) {
        return ((float)Byte.toUnsignedInt(b)) / 255;
    }

    private static byte[] getLabels(FileInputStream testFile) throws IOException {
        final var testImages = new GZIPInputStream(testFile);
        final var testImagesData = new DataInputStream(testImages);
        final var magic = testImagesData.readInt();

        if (magic != 2049) {
            throw new IOException("Magick is invalid: " + magic);
        }

        final var count = testImagesData.readInt();

        return testImagesData.readNBytes(count);
    }
}
