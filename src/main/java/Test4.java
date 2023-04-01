import neural.*;
import neural.optimizer.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;

/**
 * Тест сходимости с разными алгоритмами оптимизации
 */
public class Test4 extends TestBase{

    private static final int EPOCHS = 100;
    private static final float INITIAL_SPEED = 0.0035f;
    public static final int BATCH_SIZE = 100;

    public static void main(String[] args) throws RuntimeException {
        try (
                var testImagesFile = new FileInputStream("src/main/resources/t10k-images-idx3-ubyte.gz");
                var testLabelsFile = new FileInputStream("src/main/resources/t10k-labels-idx1-ubyte.gz");
                var trainImagesFile = new FileInputStream("src/main/resources/train-images-idx3-ubyte.gz");
                var trainLabelsFile = new FileInputStream("src/main/resources/train-labels-idx1-ubyte.gz")
        ) {
            var start = System.currentTimeMillis();

            var testImages = getImagesBatch(testImagesFile);
            var testLabels = getLabels(testLabelsFile);
            var trainImages = getImagesBatch(trainImagesFile);
            var trainLabels = getLabels(trainLabelsFile);

            var loaded = System.currentTimeMillis() - start;

            System.out.println("Files loaded " + loaded + " ms");

            var result = trainImages.length;

            for (var i = 0; i <= 6; i++) {
                var speed = INITIAL_SPEED;
                for (var j = 0; j <= 6; j++) {
                    var a = 80 * Math.pow(2, i);
                    var dropoutInput = 0.35f;
                    var random = new SecureRandom(new byte[]{3});
                    var dropoutInputAlgo = new Dropout.Zero(new Random(random.nextLong()), dropoutInput);
                    var dropoutA = 0.015f * j * (float)Math.pow(1.2, j + i);

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

                    if (trainRate > 10 && speed > 0.005f * INITIAL_SPEED) {
                        speed *= 0.7f;
                        j--;
                    }

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

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();

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
                    var r = p.trainBatch(imagesBuffer, labelsBuffer, speed);
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

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed + ". Test error rate is: " + testRate * 100 + "%. (" + testRateAvg * 100 + "%). bias: " + bias);

            if (fail == 0 || failRate > 0.6) {
                break;
            }
        }

        return fail;
    }
}
