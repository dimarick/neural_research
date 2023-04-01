import java.io.*;
import java.util.Collections;
import java.util.LinkedList;

public class Test0 extends TestBase {

    public static final float INITIAL_SPEED = 0.03f;
    public static final float SPEED_SCALE = 0.6f;

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

            var fail = test(testImages, testLabels, p, System.out);

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

        for (var epoch = 0; epoch < 40 && (fail / testImages.length) > 0.081; epoch++) {
            var epochStart = System.currentTimeMillis();

            if (prevFail > fail) {
                speed /= SPEED_SCALE;
            }

            prevFail = fail;

            Collections.shuffle(order);

            for (var i : order) {
                byte label = trainLabels[i];
                var target = createTargetForLabel(label);
                p.train(trainImages[i], target, speed);
            }

            speed *= SPEED_SCALE;

            fail = test(testImages, testLabels, p, new PrintStream(PrintStream.nullOutputStream()));

            var epochTime = System.currentTimeMillis() - epochStart;

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + (fail / testImages.length) * 100 + "%");
        }
    }
}
