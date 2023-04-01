import neural.RumelhartPerceptron;

import java.io.*;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

public class TestBase {
    protected static int getAnswer(float[] result) {
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

    protected static float getDistanceEst(float[] result, float[] target) {
        var d = 0.0f;

        for (var i = 0; i < result.length; i++) {
            float d1 = result[i] - target[i];
            d += d1 * d1;
        }

        return d;
    }

    protected static float[] createTargetForLabel(byte label) {
        var result = new float[10];

        result[label] = 1.0f;

        return result;
    }

    protected static float[][] getImages(FileInputStream testFile) throws IOException {
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
                    images[image][i] = TestBase.normalize(bytes[i]);
                }

                image++;
            }
        } catch (EOFException e) {
            System.out.println("EOF: " + e.getMessage());
        }

        return images;
    }

    private static float normalize(byte b) {
        return ((float) Byte.toUnsignedInt(b)) / 255;
    }

    protected static byte[] getLabels(FileInputStream testFile) throws IOException {
        final var testImages = new GZIPInputStream(testFile);
        final var testImagesData = new DataInputStream(testImages);
        final var magic = testImagesData.readInt();

        if (magic != 2049) {
            throw new IOException("Magick is invalid: " + magic);
        }

        final var count = testImagesData.readInt();

        return testImagesData.readNBytes(count);
    }


    protected static float[] getImagesBatch(FileInputStream testFile) throws IOException {
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

    protected static float test(float[][] testImages, byte[] testLabels, SingleLayerPerceptron p, PrintStream output) {

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

    protected static float test(float[][] testImages, byte[] testLabels, RosenblattPerceptron p) {

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

    protected static float testBatch(float[] testImages, byte[] testLabels, RumelhartPerceptron p) {
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
}
