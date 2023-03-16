package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

import java.util.Arrays;

public class Optimizer {
    public interface Interface {
        float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta);
    }

    public static class StochasticGradientDescent implements Interface {
        private LayerStaticMemory[] memory;
        private record LayerStaticMemory(VectorF32 diff, VectorF32 error, MatrixF32 errorMatrix, VectorF32 target, VectorF32 gradient, float[] loss, int[] i) {}

        public float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta) {
            if (memory == null) {
                memory = new LayerStaticMemory[layers.length];
                for (var i = 0; i < layers.length; i++) {
                    int size = layers[i].size;
                    var err = new float[size];
                    memory[i] = new LayerStaticMemory(new VectorF32(new float[size]), new VectorF32(err), new MatrixF32(1, err.length, err), new VectorF32(new float[size]), new VectorF32(new float[size]), new float[1], new int[1]);
                }
            }

            int outLayerId = layerResults.length - 1;
            var result = layerResults[outLayerId];
            var outputLayer = layers[outLayerId];

            var outMemory = memory[outLayerId];

            outputLayer.activation.diff(result, outMemory.diff);
            System.arraycopy(result.getData(), 0, outMemory.error.getData(), 0, target.getData().length);

            Ops.add(target.getData(), outMemory.error.getData(), -1.0f);

            var loss = outputLayer.loss.apply(target, result);

            if (loss == 0) {
                return 0;
            }

            for (var i = layers.length - 2; i > 0; i--) {
                var layer = layers[i];
                var currentResult = layerResults[i];
                var mem = memory[i];
                mem.i[0] = i;

                float[] currentResultData = currentResult.getData();
                System.arraycopy(currentResultData, 0, mem.target.getData(), 0, currentResultData.length);

                layer.activation.diff(currentResult, mem.diff).getData();
                Ops.multiple(memory[i + 1].errorMatrix, layers[i + 1].weights, mem.errorMatrix, 1.0f, 0.0f).getData();
                Ops.multipleBand(mem.error, mem.diff, mem.gradient, 1.0f, 0.0f);
                Ops.add(mem.gradient.getData(), mem.target.getData(), 1.0f);

                mem.loss[0] = layer.loss.apply(mem.target, currentResult);
            }

            Ops.multipleBand(outMemory.error, outMemory.diff, outMemory.gradient, 1.0f, 0.0f);

            Arrays.stream(memory).parallel().forEach(mem -> {
                var i = mem.i[0];
                if (i == 0) {
                    return;
                }

                Ops.multiple(mem.gradient, layerResults[i - 1], layers[i].weights, -eta * mem.loss[0] * layers[i].dropout.getRate(layerResults[i - 1]), 1.0f);
            });

            return loss;
        }
    }
}
