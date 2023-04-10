import linear.VectorF32;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Collections;
import java.util.LinkedList;

/**
 * Определение наилучшей скорости для разных A-size
 */
public class Test1 extends TestBase {

    private static final int EPOCHS = 300;
    private static final float INITIAL_SPEED = 0.005f;

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

            var speed = INITIAL_SPEED;

            for (var i = 5; i < 30; i++) {
                var a = 125 * Math.pow(2, i);
                System.out.println("Starting test with speed " + speed + "(" + a + ")");
                var p = new RosenblattPerceptron(28 * 28, 10, (int) (a), new SecureRandom(new byte[]{3}));
                result = train(testImages, testLabels, trainImages, trainLabels, speed, p);

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

    private static int train(float[][] testImages, byte[] testLabels, float[][] trainImages, byte[] trainLabels, float speed, RosenblattPerceptron p) {
        var order = new LinkedList<Integer>();

        for (var i = 0; i < trainImages.length; i++) {
            order.add(i);
        }

        var layer1 = new VectorF32[trainImages.length];

        for (var i = 0; i < trainImages.length; i++) {
            layer1[i] = p.evalLayer1(trainImages[i]);
        }

        var fail = 0;
        var prevFail = trainImages.length;
        var failRate = 0.1f;
        var bestResult = prevFail;

        var speedScale = 1f;
        var speedDecayTime = 80f;
        var speedDecayStart = 0;

        for (var epoch = 0; epoch < EPOCHS; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();

            if ((epoch - speedDecayStart) > speedDecayTime) {
                speedScale *= 0.5f;
                speedDecayTime *= 0.9;
                speedDecayStart = epoch;
            }

            // Перемешивание образцов ускоряет сходимость сети
            Collections.shuffle(order);

            for (var i : order) {
                byte label = trainLabels[i];
                var target = createTargetForLabel(label);
                var r = p.trainLayer2(layer1[i], target, speed * speedScale, 0);
                if (getAnswer(r) != label) {
                    fail++;
                }
            }

            failRate = ((float) fail / trainImages.length);

            var testFail = test(testImages, testLabels, p);

            var epochTime = System.currentTimeMillis() - epochStart;
            float testRate = (testFail / testImages.length);

            if (fail < bestResult) {
                bestResult = fail;
            }

            if ((float)bestResult / fail < 0.7) {
                break;
            }

            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed * speedScale + "). Test error rate is: " + testRate * 100 + "%. bias: " + (failRate * 100 - testRate * 100));
        }

        return fail;
    }
}
