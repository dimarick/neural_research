import java.io.*;
import java.util.Collections;
import java.util.LinkedList;

/**
 * Тест сходимости Розенблатта
 */
public class Test0 extends TestBase {

    public static final float INITIAL_SPEED = 0.02f;

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

            var p = new SingleLayerPerceptron(28 * 28, 10, 1);

            train(testImages, testLabels, trainImages, trainLabels, p);

            var testStart = System.currentTimeMillis();

            var fail = test(testImages, testLabels, p);

            System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + (fail / testImages.length) * 100 + "%");

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void train(float[][] testImages, byte[] testLabels, float[][] trainImages, byte[] trainLabels, SingleLayerPerceptron p) {
        var prevFail = (float) trainImages.length + 1;
        var fail = prevFail - 1;
        var speed = INITIAL_SPEED;
        var order = new LinkedList<Integer>();

        for (var i = 0; i < trainImages.length; i++) {
            order.add(i);
        }

        var speedScale = 1f;
        var speedDecayTime = 15f;
        var speedDecayStart = 0;

        for (var epoch = 0; epoch < 100; epoch++) {
            var epochStart = System.currentTimeMillis();
            if ((epoch - speedDecayStart) > speedDecayTime) {
                speedScale *= 0.5f;
                speedDecayStart = epoch;
            }

            Collections.shuffle(order);

            for (var i : order) {
                byte label = trainLabels[i];
                var target = createTargetForLabel(label);
                p.train(trainImages[i], target, speed * speedScale);
            }

            fail = test(testImages, testLabels, p);
            var trainFail = test(trainImages, trainLabels, p);

            var epochTime = System.currentTimeMillis() - epochStart;

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + (trainFail / trainImages.length) * 100 + "%. Test error rate is: " + (fail / testImages.length) * 100 + "%. speed was: " + speed * speedScale);
        }
    }
}
