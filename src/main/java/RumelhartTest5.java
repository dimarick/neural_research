import neural.Activation;
import neural.Dropout;
import neural.Regularization;
import neural.RumelhartPerceptron;
import neural.optimizer.*;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;
import java.util.zip.GZIPInputStream;

/**
 * Тест лучшего результата
 */
public class RumelhartTest5 {

    private static final int EPOCHS = 2000;
    private static final float INITIAL_SPEED = 0.0001f;
    public static final int BATCH_SIZE = 100;

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

            for (var i = 0; i <= 8; i++) {
                for (var j = 0; j <= 0; j++) {
                    var a = 40 * Math.pow(2, i);
                    var dropoutInput = 0.35f;
                    var random = new SecureRandom(new byte[]{3});
                    var dropoutInputAlgo = new Dropout.Zero(new Random(random.nextLong()), dropoutInput);
                    var dropoutA = switch ((int)a) {
                        case 40 -> 0.01f;
                        case 80 -> 0.01f;
                        case 160 -> 0.05f;
                        case 320 -> 0.05f;
                        case 640 -> 0.05f;
                        case 1280 -> 0.45f;
                        case 2560 -> 0.55f;
                        case 5120 -> 0.75f;
                        default -> 0;
                    };

                    var speed = switch ((int)a) {
                        case 40 -> 0.0008f;
                        case 80 -> 0.0006f;
                        case 160 -> 0.0005f;
                        case 320 -> 0.0003f;
                        case 640 -> 0.0002f;
                        case 1280 -> 0.00015f;
                        case 2560 -> 0.00004f;
                        case 5120 -> 0.00001f;
                        default -> 0.001f;
                    };

                    var rAlgo = new Regularization.Ridge(1e-3f);

                    var p = new RumelhartPerceptron(random, new AdamBatch())
                            .addLayer(28 * 28)
                            .set(new Activation.LeakyReLU())
                            .set(dropoutInputAlgo)
                            .parent()

                            .addLayer((int)a)
                            .set(new Activation.LeakyReLU())
                            .set(new Dropout.Zero(new Random(random.nextLong()), dropoutA))
                            .set(rAlgo).parent()

                            .addLayer((int)a)
                            .set(new Activation.LeakyReLU())
                            .set(new Dropout.Zero(new Random(random.nextLong()), dropoutA))
                            .set(rAlgo).parent()

                            .addLayer(10)
                            .set(new Activation.SoftmaxStable())
                            .set(rAlgo)
                            .parent();

                    System.out.println("Starting test with speed " + speed + "(" + a + ", " + 1e-6f + "), volume " + p.volume() + ", dropout I: " + dropoutInputAlgo.getClass().getSimpleName() + ", " + dropoutInput + ", A:" + dropoutA);

                    result = train(testImages, testLabels, trainImages, trainLabels, speed, p);

                    var testStart = System.currentTimeMillis();

                    var fail = testBatch(testImages, testLabels, p);

                    var trainRate = ((float)result / trainLabels.length) * 100;

//                    if (trainRate > 10 && speed > 0.00005f * INITIAL_SPEED) {
//                        speed *= 0.7f;
//                        j--;
//                    }

                    var testRate = (fail / testLabels.length) * 100;
                    System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + trainRate + "% " + ". Test Error rate is: " + testRate + "%");
                }
            }

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int train(float[] testImages, byte[] testLabels, float[] trainImages, byte[] trainLabels, float speed, RumelhartPerceptron p) {
        var order = new LinkedList<Integer>();

        int imageSize = p.inputSize();

        int imageCount = trainImages.length / imageSize;
        int trainSize = 60000;//imageCount;
        int testSize = testImages.length / imageSize;//imageCount;

        for (var i = 0; i < imageCount; i++) {
            order.add(i);
        }

        var fail = 0;
        var failRate = 0.1f;

        var batchSize = BATCH_SIZE;

        var imagesBuffer = new float[imageSize * batchSize];
        var labelsBuffer = new float[10 * batchSize];
        var testRateAvg = -1f;
        var speedScale = 1f;
        var speedDecayTime = 80f;
        var speedDecayStart = 0;

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();
            if ((epoch - speedDecayStart) > speedDecayTime) {
                speedScale *= 0.5f;
                speedDecayTime *= 0.7;
                speedDecayStart = epoch;
            }

            // Перемешивание образцов ускоряет сходимость сети
            Collections.shuffle(order);

            var j = 0;
            for (var i : order.subList(0, trainSize)) {
                byte label = trainLabels[i];
                System.arraycopy(createTargetForLabel(label), 0, labelsBuffer, j * 10, 10);
                System.arraycopy(trainImages, i * imageSize, imagesBuffer, j * imageSize, imageSize);
                j++;

                if (j >= batchSize) {
                    j = 0;
                    var r = p.trainBatch(imagesBuffer, labelsBuffer, speed * speedScale);
                    for (var k = 0; k < batchSize; k++) {
                        if (getAnswer(Arrays.copyOfRange(r, k * 10, (k + 1) * 10)) != getAnswer(Arrays.copyOfRange(labelsBuffer, k * 10, (k + 1) * 10))) {
                            fail++;
                        }
                    }
                }
            }

            failRate = ((float) fail / trainSize);

            var testFail = testBatch(testImages, testLabels, p);

            var epochTime = System.currentTimeMillis() - epochStart;
            float testRate = (testFail / testSize);

            float bias = testRate * 100 - failRate * 100;

            testRateAvg = testRateAvg == -1 ? testRate * 0.5f : 0.1f * testRate + testRateAvg * 0.9f;

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed * speedScale + ". Test error rate is: " + testRate * 100 + "%. (" + testRateAvg * 100 + "%). bias: " + bias);

            if (fail == 0 || (failRate > 0.08 && epoch > 10) || speedScale < 1e-10) {
                break;
            }
        }

        return fail;
    }

    private static float testBatch(float[] testImages, byte[] testLabels, RumelhartPerceptron p) {
        var fail = 0.0f;
        var results = p.evalBatch(testImages);

        for (var i = 0; i < testLabels.length; i++) {
            byte label = testLabels[i];

            int answer = getAnswer(Arrays.copyOfRange(results, 10 * i, 10 * (i + 1)));

            if (answer != label) {
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

    private static float[] getImages(FileInputStream testFile) throws IOException {
        final var testImages = new GZIPInputStream(testFile);
        final var testImagesData = new DataInputStream(testImages);
        final var magic = testImagesData.readInt();

        if (magic != 2051) {
            throw new IOException("Magick is invalid: " + magic);
        }

        final var count = testImagesData.readInt();
        final var rows = testImagesData.readInt();
        final var columns = testImagesData.readInt();
        int size = rows * columns;
        final var images = new float[count * size];

        int image = 0;

        try {
            while (image < count) {
                byte[] bytes = testImagesData.readNBytes(rows * columns);

                for (var i = 0; i < bytes.length; i++) {
                    images[image * size + i] = normalize(bytes[i]);
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
