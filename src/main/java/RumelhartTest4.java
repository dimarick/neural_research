import neural.*;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class RumelhartTest4 {

    private static final int EPOCHS = 50;
    private static final float INITIAL_SPEED = 0.2f;

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

            for (var i = 0; i <= 6; i++) {
                var speed = INITIAL_SPEED;
                for (var j = 0; j <= 20; j++) {
                    var a = 320 * Math.pow(2, i);
                    var b = 50 * Math.pow(2, j);
                    var dropoutInput = 0.18f;
                    SecureRandom random = new SecureRandom(new byte[]{3});
                    var dropoutInputAlgo = new Dropout.Zero(new Random(random.nextLong()), dropoutInput);
                    var dropoutA = 0.0f * j;

                    var rAlgo = new Regularization.ElasticNet(1e-6f);

                    System.out.println("Starting test with speed " + speed + "(" + a + ", " + 1e-6f + "), " + rAlgo.getClass().getSimpleName() + ", dropout I: " + dropoutInputAlgo.getClass().getSimpleName() + ", A:" + dropoutA);

                    var p = new RumelhartPerceptron(random, new Optimizer.StochasticGradientDescent())
                            .addLayer(28 * 28)
                            .set(new Activation.ReLU())
                            .set(dropoutInputAlgo)
                            .parent()

                            .addLayer((int)a)
                            .set(new Activation.Linear())
                            .set(new Dropout.Zero(new Random(random.nextLong()), dropoutA))
                            .set(rAlgo).parent()

                            .addLayer(10)
                            .set(rAlgo)
                            .parent();

                    result = train(testImages, testLabels, trainImages, trainLabels, speed, p);

                    var testStart = System.currentTimeMillis();

                    var fail = test(testImages, testLabels, p);

                    var trainRate = ((float)result / trainImages.length) * 100;

                    if (trainRate > 10 && speed > 0.005f) {
                        speed *= 0.7f;
                        j--;
                    }

                    var testRate = (fail / testImages.length) * 100;
                    System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + trainRate + "% " + ". Test Error rate is: " + testRate + "%");
                }
            }

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int train(float[][] testImages, byte[] testLabels, float[][] trainImages, byte[] trainLabels, float speed, RumelhartPerceptron p) {
        var order = new LinkedList<Integer>();

        int trainSize = 60000;//trainImages.length;
        for (var i = 0; i < trainImages.length; i++) {
            order.add(i);
        }

        var fail = 0;
        var failRate = 0.1f;

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();

            // Перемешивание образцов ускоряет сходимость сети
            Collections.shuffle(order);

            for (var i : order.subList(0, trainSize)) {
                byte label = trainLabels[i];
                var target = createTargetForLabel(label);
                var r = p.train(trainImages[i], target, speed);
                if (getAnswer(r) != label) {
                    fail++;
                }
            }

            failRate = ((float) fail / trainSize);

            var testFail = test(testImages, testLabels, p);

            var epochTime = System.currentTimeMillis() - epochStart;
            float testRate = (testFail / testImages.length);

            float bias = testRate * 100 - failRate * 100;

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed + ". Test error rate is: " + testRate * 100 + "%. bias: " + bias);

            if (fail == 0 || failRate > 0.6) {
                break;
            }
        }

        return fail;
    }

    private static float test(float[][] testImages, byte[] testLabels, RumelhartPerceptron p) {

        var fail = 0.0f;

        for (var i = 0; i < testImages.length; i++) {
            byte label = testLabels[i];
            var result = p.eval(testImages[i]);

            int answer = getAnswer(result);

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
