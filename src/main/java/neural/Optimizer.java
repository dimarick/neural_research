package neural;

import linear.Ops;
import linear.VectorF32;

public class Optimizer {
    public interface Interface {
        float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta);
    }

    public static class StochasticGradientDescent implements Interface {
        public float apply(Layer[] layers, VectorF32[] layerResults, VectorF32 target, float eta) {
            var result = layerResults[layerResults.length - 1];
            var outputLayer = layers[layers.length - 1];

            var diff = outputLayer.activation.diff(result);
            var error = Ops.subtract(result.getData(), target.getData());

            var loss = outputLayer.loss.apply(target, result);

            if (loss == 0) {
                return 0;
            }

            var nextLayerResult = result.getData();
            var nextLayer = outputLayer.weights.getData();
            var nextLayerLoss = error;
            var currentResult = layerResults[layers.length - 2];

            for (var i = layers.length - 2; i > 0; i--) {
                var layer = layers[i];

                float[] currentResultData = currentResult.getData();
                var layerTarget = currentResultData.clone();
                var layerError = currentResultData.clone();

                var currentResultDiff = layer.activation.diff(currentResult).getData();

                for (var j = 0; j < layerTarget.length; j++) {
                    var sum = 0.0f;
                    for (var k = 0; k < nextLayerResult.length; k++) {
                        sum += nextLayerLoss[k] * nextLayer[k * currentResultData.length + j];
                    }

                    layerTarget[j] += currentResultDiff[j] * sum;
                    layerError[j] = currentResultDiff[j] * sum;
                }

                var loss2 = layer.loss.apply(new VectorF32(layerTarget), currentResult);

                Ops.multiple(new VectorF32(layerError), layerResults[i - 1], layer.weights, -eta * loss2 * layer.dropout.getRate(layerResults[i - 1]), 1.0f);

                nextLayerResult = currentResultData;
                nextLayer = layer.weights.getData();
                nextLayerLoss = layerError;
                currentResult = layerResults[i];
            }

            var delta = new float[error.length];

            for (var i = 0; i < error.length; i++) {
                delta[i] = -eta * loss * error[i] * diff.getData()[i];
            }

            Ops.multiple(new VectorF32(delta), layerResults[layerResults.length - 2], outputLayer.weights, 1.0f, 1.0f).getData();

            return loss;
        }
    }
}
