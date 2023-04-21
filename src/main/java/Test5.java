import neural.Activation;
import neural.Dropout;
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
 * Тест лучшего результата с dropout из предыдущих бенчмарков
 */
public class Test5 extends TestBase {

    private static final int EPOCHS = 600;
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

            for (var i = 0; i <= 8; i++) {
                for (var j = 0; j <= 6; j++) {
                    for (var k = 0; k <= 4; k++) {
                        var a = 10 * Math.pow(2, i);

                        var dropoutInput = switch ((int)a) {
                            case 10 -> 0.05f;
                            case 20 -> 0.1f;
                            case 40 -> 0.2f;
                            case 80 -> 0.26f;
                            case 160 -> 0.3f;
                            case 320 -> 0.33f;
                            default -> 0.33f;
                        } * 0.8f;

                        var random = new SecureRandom(new byte[]{(byte)k});
                        var dropoutInputAlgo = new Dropout.Zero(new Random(random.nextLong()), dropoutInput);

                        var speed = switch ((int)a) {
                            case 10 -> 0.0003f;
                            case 20 -> 0.0005f;
                            case 40 -> 0.001f;
                            case 80 -> 0.001f;
                            case 160 -> 0.001f;
                            case 320 -> 0.001f;
                            case 640 -> 0.001f;
                            case 1280 -> 0.001f;
                            case 2560 -> 0.001f;
                            default -> 0.001f;
                        };

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
                            case 2 -> 1;
                            case 3 -> 100f;
                            case 4 -> 4f;
                            case 5 -> 0.01f;
                            default -> 1f;
                        };

                        var dropoutA = switch ((int)a) {
                            case 80 -> 0.13f;
                            case 160 -> 0.16f;
                            case 320 -> 0.16f;
                            case 640 -> 0.08f;
                            case 1280 -> 0.17f;
                            case 2560 -> 0.1f;
                            default -> 0.0f;
                        };

                        if (optimizer instanceof AdaDelta && a >= 1280) {
                            speedOptimizerScale *= 0.3f;
                        }

                        var dropout = new Dropout.Zero(new Random(random.nextLong()), dropoutA);

                        var p = new FeedForwardNeuralNetwork(random, optimizer)
                                .addLayer(28 * 28)
                                .set(new Activation.LeakyReLU())
                                .set(dropoutInputAlgo)
                                .parent()

                                .addLayer((int)a)
                                .set(new Activation.LeakyReLU())
                                .set(dropout).parent()

                                .addLayer((int)a)
                                .set(new Activation.LeakyReLU())
                                .set(dropout).parent()

                                .addLayer(10)
                                .set(new Activation.SoftmaxStable())
                                .parent();

                        System.out.println("Starting test with speed " + speed * speedOptimizerScale + "(" + a + "), volume " + p.volume() + ", Opt: " + optimizer.getClass().getSimpleName());

                        result = train(testImages, testLabels, trainImages, trainLabels, speed * speedOptimizerScale, p);

                        var testStart = System.currentTimeMillis();

                        var fail = testBatch(testImages, testLabels, p);

                        var trainRate = ((float)result / trainLabels.length) * 100;

                        var testRate = (fail / testLabels.length) * 100;
                        System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + trainRate + "% " + ". Test Error rate is: " + testRate + "%");
                    }
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
        var bestDO = 0f;

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();
            if ((epoch - speedDecayStart) > speedDecayTime) {
                speedScale *= 0.5f;
//                speedDecayTime *= 0.9;
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

            testRateAvg = testRateAvg == -1 ? testRate : 0.2f * testRate + testRateAvg * 0.8f;

            if (testRateAvg < bestTestRateAvg) {
                bestTestRateAvg = testRateAvg;
                bestEpoch = epoch;
            }

            if (failRate < bestTrainRate) {
                bestTrainEpoch = epoch;
                bestTrainRate = failRate;
            }

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed * speedScale + ". Test error rate is: " + testRate * 100 + "%. (" + testRateAvg * 100 + "%)");

            if ((fail == 0 && epoch > 150) || (testRateAvg - bestTestRateAvg > 0.03 && epoch > 20) || speedScale < 1e-10) {
                break;
            }
        }

        System.out.println("Best test result epoch is " + bestEpoch + ". Test error rate is: " + bestTestRateAvg * 100 + " bestDO: " + bestDO);
        System.out.println("Best train result epoch is " + bestTrainEpoch + ". Error rate is: " + bestTrainRate * 100);

        return fail;
    }
}
