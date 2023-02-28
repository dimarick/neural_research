import linear.matrix.MatrixF32Interface;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class RosenblattTest4 {

    private static final float INITIAL_SPEED = 5.0f;
    private static final float SPEED_SCALE = 0.9f;

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

            for (var i = 0; i < 30; i++) {
                var a = 32000 * Math.pow(2, i);
                System.out.println("Starting test with speed " + speed + "(" + a + ")");
                var p = new RosenblattPerceptron(28 * 28, 10, (int) (a), new SecureRandom(new byte[]{2}));
                result = train(testImages, testLabels, trainImages, trainLabels, speed, p);

                var testStart = System.currentTimeMillis();

                var fail = test(testImages, testLabels, p);

                System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + (result / trainImages.length) * 100 + "% " + ". Test Error rate is: " + (fail / testImages.length) * 100 + "% " + speed);
            }

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static int train(float[][] testImages, byte[] testLabels, float[][] trainImages, byte[] trainLabels, float speed, PerceptronInterface p) {
        var order = new LinkedList<Integer>();

        for (var i = 0; i < trainImages.length; i++) {
            order.add(i);
        }

        var layer1 = new MatrixF32Interface[trainImages.length];

        for (var i = 0; i < trainImages.length; i++) {
            layer1[i] = p.evalLayer1(trainImages[i]);
        }

        var fail = 0;
        var prevFail = trainImages.length;
        var prevWeights = p.getAssocLayer();
        var failRate = 1.0f;

        for (var epoch = 0; epoch < 50; epoch++) {
            fail = 0;
            var epochStart = System.currentTimeMillis();

            // Перемешивание образцов ускоряет сходимость сети
            Collections.shuffle(order);

            for (var i : order) {
                byte label = trainLabels[i];
                var target = createTargetForLabel(label);
                var r = p.trainLayer2(layer1[i], target, speed, true);
                if (getAnswer(r) != label) {
                    fail++;
                }

            }

            failRate = ((float) fail / trainImages.length);

            var testFail = test(testImages, testLabels, p);

            var epochTime = System.currentTimeMillis() - epochStart;
            System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + failRate * 100 + "%. speed was: " + speed + "%. Test error rate is: " + (testFail / testImages.length) * 100 + "%");

            if (fail < prevFail) {
                prevWeights = p.getAssocLayer();
                prevFail = fail;
                if (speed > INITIAL_SPEED * 0.05f) {
                    speed *= SPEED_SCALE;
                }
            } else if ((float)fail / prevFail < 0.08f) {
                speed *= new Random().nextFloat(0.9f, 1.25f);
            } else {
                p.setAssocLayer(prevWeights);
                speed *= new Random().nextFloat(0.87f, 1.1f);
            }
        }

//        var testStart = System.currentTimeMillis();

//        var testFail = test(testImages, testLabels, p);

//        var testTime = System.currentTimeMillis() - testStart;

//        System.out.println("test done. " + testTime + " ms. Error rate is: " + (testFail / testImages.length) * 100 + "%");

        return fail;
    }

    private static float test(float[][] testImages, byte[] testLabels, PerceptronInterface p) {

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
