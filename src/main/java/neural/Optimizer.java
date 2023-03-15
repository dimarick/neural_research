package neural;

import linear.MatrixF32;
import linear.Ops;
import linear.VectorF32;

public class Optimizer {
    public interface Interface {
        float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta);
    }

    public static class StochasticGradientDescent implements Interface {
        private LayerStaticMemory[] memory;
        private record LayerStaticMemory(VectorF32 diff, VectorF32 error, MatrixF32 errorMatrix, VectorF32 target, VectorF32 gradient) {}

        public float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta) {
            if (memory == null) {
                memory = new LayerStaticMemory[layers.length];
                for (var i = 0; i < layers.length; i++) {
                    int size = layers[i].size;
                    var err = new float[size];
                    memory[i] = new LayerStaticMemory(new VectorF32(new float[size]), new VectorF32(err), new MatrixF32(1, err.length, err), new VectorF32(new float[size]), new VectorF32(new float[size]));
                }
            }

            int outLayerId = layerResults.length - 1;
            var result = layerResults[outLayerId];
            var outputLayer = layers[outLayerId];

            var outMemory = memory[outLayerId];

            var diff = outputLayer.activation.diff(result, outMemory.diff);
            System.arraycopy(result.getData(), 0, outMemory.error.getData(), 0, target.getData().length);

            var error = Ops.add(target.getData(), outMemory.error.getData(), -1.0f);

            var loss = outputLayer.loss.apply(target, result);

            if (loss == 0) {
                return 0;
            }

            for (var i = layers.length - 2; i > 1; i--) {
                var layer = layers[i];
                var currentResult = layerResults[i];
                var mem = memory[i];

                float[] currentResultData = currentResult.getData();
                System.arraycopy(currentResultData, 0, mem.target.getData(), 0, currentResultData.length);

                var layerTarget = mem.target.getData();

                var currentResultDiff = layer.activation.diff(currentResult, mem.diff).getData();

                var layerError = Ops.multiple(memory[i + 1].errorMatrix, layers[i + 1].weights, mem.errorMatrix, 1.0f, 0.0f).getData();

                var gradient = mem.gradient.getData();

                for (var j = 0; j < layerError.length; j++) {
                    gradient[j] = layerError[j] * currentResultDiff[j];
                    layerTarget[j] += gradient[j];
                }

                var loss2 = layer.loss.apply(mem.target, currentResult);

                Ops.multiple(mem.gradient, layerResults[i - 1], layer.weights, -eta * loss2 * layer.dropout.getRate(layerResults[i - 1]), 1.0f);
            }

            var gradient = outMemory.gradient.getData();

            for (var i = 0; i < error.length; i++) {
                gradient[i] = -eta * loss * error[i] * diff.getData()[i];
            }

            Ops.multiple(outMemory.gradient, layerResults[outLayerId - 1], outputLayer.weights, 1.0f, 1.0f).getData();

            return loss;
        }
    }
}
