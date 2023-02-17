import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.zip.GZIPInputStream;

public class Main {

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

            var prevFail = (float)trainImages.length + 1;
            var fail = prevFail - 1;
            var speed = INITIAL_SPEED;
            var order = new LinkedList<Integer>();

            for (var i = 0; i < trainImages.length; i++) {
                order.add(i);
            }

            for (var epoch = 0; epoch < 40 && (fail / testImages.length) > 0.081; epoch++) {
                var epochStart = System.currentTimeMillis();

                if (prevFail > fail) {
                    speed *= 1.2;
                }

                prevFail = fail;

                Collections.shuffle(order);

                for (var i : order) {
                    byte label = trainLabels[i];
                    var result = p.eval(trainImages[i]);
                    var target = createTargetForLabel(label);
                    p.train(trainImages[i], result, target, speed);
                }

                speed *= SPEED_SCALE;

                fail = test(testImages, testLabels, p, new PrintStream(PrintStream.nullOutputStream()));

                var epochTime = System.currentTimeMillis() - epochStart;

                System.out.println("epoch is " + epoch + " done. " + epochTime + " ms. Error rate is: " + (fail / testImages.length) * 100 + "%");
            }

            var testStart = System.currentTimeMillis();

            fail = test(testImages, testLabels, p, System.out);

            System.out.println("test is done. " + (System.currentTimeMillis() - testStart) + " ms. Error rate is: " + (fail / testImages.length) * 100 + "%");

            System.out.println("Success");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static float test(float[][] testImages, byte[] testLabels, SingleLayerPerceptron p, PrintStream output) {

        var fail = 0.0f;

        for (var i = 0; i < testImages.length; i++) {
            byte label = testLabels[i];
            var result = p.eval(testImages[i]);
            var target = createTargetForLabel(label);

            int answer = getAnswer(result);

            if (answer != label) {
                output.println("Wrong!!!  i = " + i + ";" + getDistanceEst(result, target) + ", answer is " + answer + "(" + result[answer] + ") != " + label + "    " + Arrays.toString(result));
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

    private static float getDistanceEst(float[] result, float[] target) {
        var d = 0.0f;

        for (var i = 0; i < result.length; i++) {
            float d1 = result[i] - target[i];
            d += d1 * d1;
        }

        return d;
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
