import neural.Activation;
import neural.Dropout;
import neural.Regularization;
import neural.FeedForwardNeuralNetwork;
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
public class Test3 extends TestBase {

    private static final int EPOCHS = 2000;
    public static final int BATCH_SIZE = 200;

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

            for (var i = 0; i <= 9; i++) {
                for (var j = 0; j <= 6; j++) {
                    var a = 10 * Math.pow(2, i);
                    var dropoutInput = 0f;
                    var random = new SecureRandom(new byte[]{3});
                    var dropoutInputAlgo = new Dropout.Zero(new Random(random.nextLong()), dropoutInput);

                    var speed = switch ((int)a) {
                        case 10 -> 0.001f;
                        case 20 -> 0.0008f;
                        case 40 -> 0.0006f;
                        case 80 -> 0.0004f;
                        case 160 -> 0.00025f;
                        case 320 -> 0.0002f;
                        case 640 -> 0.0002f;
                        case 1280 -> 0.00015f;
                        case 2560 -> 0.00004f;
                        case 5120 -> 0.00001f;
                        default -> 0.001f;
                    };

                    var rAlgo = new Regularization.ElasticNet(1e-5f);

                    var dropout = new Dropout.Zero(new Random(random.nextLong()), 0);
                    var optimizer = switch (j) {
                        case 0 -> new SGD();
                        case 1 -> new Momentum(0.95f);
                        case 2 -> new Nesterov(0.8f);
                        case 3 -> new AdaGrad();
                        case 4 -> new RMSProp(0.99f);
                        case 5 -> new AdaDelta(0.999f);
                        default -> new Adam();
                    };

                    var speedOptimizerScale = switch (j) {
                        case 0 -> 2;
                        case 1 -> 2;
                        case 2 -> 3;
                        case 3 -> 100f;
                        case 4 -> 4f;
                        case 5 -> 0.5f;
                        default -> 1f;
                    };

                    var p = new FeedForwardNeuralNetwork(random, optimizer)
                            .addLayer(28 * 28)
                            .set(new Activation.LeakyReLU())
                            .set(dropoutInputAlgo)
                            .parent()

                            .addLayer((int)a)
                            .set(new Activation.LeakyReLU())
                            .set(dropout)
                            .set(rAlgo).parent()

                            .addLayer((int)a)
                            .set(new Activation.LeakyReLU())
                            .set(dropout)
                            .set(rAlgo).parent()

                            .addLayer(10)
                            .set(new Activation.SoftmaxStable())
                            .set(rAlgo)
                            .parent();

                    System.out.println("Starting test with speed " + speed * speedOptimizerScale + "(" + a + ", " + 1e-6f + "), volume " + p.volume() + ", dropout I: " + optimizer.getClass().getSimpleName() + ", dropout " + dropoutInput);

                    result = train(testImages, testLabels, trainImages, trainLabels, speed * speedOptimizerScale, p);

                    var testStart = System.currentTimeMillis();

                    var fail = testBatch(testImages, testLabels, p);

                    var trainRate = ((float)result / trainLabels.length) * 100;

                    var testRate = (fail / testLabels.length) * 100;
                    System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + trainRate + "% " + ". Test Error rate is: " + testRate + "%");
                }
            }

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int train(float[] testImages, byte[] testLabels, float[] trainImages, byte[] trainLabels, float speed, FeedForwardNeuralNetwork p) {
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
        var bestTestRateAvg = 1f;
        var speedScale = 1f;
        var speedDecayTime = 80f;
        var speedDecayStart = 0;
        var bestEpoch = 0;
        var bestTrainRate = 1f;
        var bestTrainEpoch = 0;

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            if (epoch == 20) {
                speedScale *= 0.5f;
            }

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
                    var r = p.train(imagesBuffer, labelsBuffer, speed * speedScale);
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

            testRateAvg = testRateAvg == -1 ? 0.018f : 0.2f * testRate + testRateAvg * 0.8f;

            if (epoch > 10) {
                if (testRateAvg < bestTestRateAvg) {
                    bestTestRateAvg = testRateAvg;
                    bestEpoch = epoch;
                }
            } else {
                if (testRate < bestTestRateAvg) {
                    bestTestRateAvg = testRate;
                    bestEpoch = epoch;
                }
            }

            if (failRate < bestTrainRate) {
                bestTrainEpoch = epoch;
                bestTrainRate = failRate;
            }

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed * speedScale + ". Test error rate is: " + testRate * 100 + "%. (" + testRateAvg * 100 + "%)");

            if (fail == 0 || (testRateAvg - bestTestRateAvg > 0.03 && epoch > 20) || speedScale < 1e-10) {
                break;
            }
        }

        System.out.println("Best test result epoch is " + bestEpoch + ". Test error rate is: " + bestTestRateAvg * 100);
        System.out.println("Best train result epoch is " + bestTrainEpoch + ". Error rate is: " + bestTrainRate * 100);

        return fail;
    }
}
