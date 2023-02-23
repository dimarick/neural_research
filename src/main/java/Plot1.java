import org.jzy3d.analysis.AWTAbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.AWTChartFactory;
import org.jzy3d.chart.factories.IChartFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Func3D;
import org.jzy3d.plot3d.builder.SurfaceBuilder;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;
import com.jogamp.opengl.awt.GLCanvas;

/**
 * Demo an AWT chart using JOGL {@link GLCanvas}.
 *
 * @author martin
 */
public class Plot1 extends AWTAbstractAnalysis {
    final private static int[] dataTime = new int[]{2678, 2372, 2089, 2087, 2164, 2450, 2159, 2770, 2488, 2454, 2504, 2558, 2373, 2444, 2735, 2576, 2457, 2492, 2589, 2528, 2690, 2703, 2826, 2956, 2724, 2798, 2942, 2742, 2956, 3236, 3171, 3065, 3028, 3083, 3138, 3644, 3545, 3582, 3610, 3664, 3457, 3380, 3999, 3838, 4005, 4104, 4009, 3894, 3933, 4630, 4169, 4278, 4515, 4400, 4192, 4473, 4399, 4550, 4802, 4888, 4108, 4713, 5089, 5510, 4316, 4946, 4800, 4375, 5176, 5401, 9607, 9242, 10458, 9647, 8965, 9501, 8535, 13224, 14155, 13826, 14075, 12543, 13452, 14702, 16601, 16902, 18628, 18320, 21465, 21237, 19898, 19654, 19110, 21347, 23582, 24051, 25023, 25500, 21338, 21744, 23476, 24118, 24011, 26037, 26918, 24585, 24519, 25462, 26080, 26882, 27057, 27170, 27800, 27576, 28827, 29606, 27984, 29896, 29403, 31161, 30589, 30871, 30984, 32496, 33486, 33231, 35044, 33656, 35930, 35514, 36467, 35520, 36018, 39692, 38643, 39288, 38210, 39579, 39793, 40671, 43571, 42547, 42784, 42866, 42882, 44685, 44123, 47141, 52319, 52256, 51328, 54467, 59416, 64611, 66043, 59301, 58519, 64895, 63597, 63474, 62920, 69294, 65744, 57981, 58844, 59024, 59625, 58479, 68444, 69746, 64346, 63459, 73703, 75942, 71847, 68992, 65417, 66324, 71302, 69597, 68272, 71613, 70153, 71756, 73628, 74218, 74199, 71809, 69374, 65120, 64588, 65855, 71746, 78163, 76850, 71706};
    final private static float[] dataErrorTrain = new float[]{20.555f,18.451668f,17.623333f,17.031666f,16.421667f,15.889999f,15.584999f,15.33f,12.716667f,12.036667f,11.406667f,10.925f,10.605f,10.225f,13.551667f,10.596666f,9.785f,9.143333f,8.751667f,8.363334f,8.04f,12.468333f,9.028334f,8.136666f,7.631667f,7.1966662f,6.8433337f,6.6466665f,12.07f,8.3133335f,7.36f,6.725f,6.1683335f,5.8916664f,5.635f,11.776667f,7.738333f,6.6600003f,6.008333f,5.376667f,5.185f,4.8250003f,11.505f,7.283334f,6.2883334f,5.46f,5.0483336f,4.601667f,4.355f,11.181666f,7.0033336f,5.855f,5.2066665f,4.636667f,4.2266665f,3.8066666f,11.093333f,6.756666f,5.5550003f,4.846667f,4.3883333f,3.8899999f,3.5783331f,10.976667f,6.5350003f,5.23f,4.541667f,3.9133332f,3.6083333f,3.3033333f,10.426666f,5.5666666f,4.186667f,3.2933333f,2.7233334f,2.2316666f,1.9333333f,10.49f,5.3399997f,3.815f,2.8816667f,2.355f,1.8066667f,1.5866667f,10.546666f,5.2066665f,3.6983333f,2.7716665f,2.22f,1.7149999f,1.3916667f,10.581667f,5.298333f,3.6649997f,2.645f,2.1016667f,1.565f,1.285f,10.691667f,5.2316666f,3.6899998f,2.6233335f,2.07f,1.52f,1.2433333f,10.668334f,5.241667f,3.605f,2.6799998f,1.9483334f,1.4733334f,1.2116667f,10.816667f,5.3683333f,3.6983333f,2.645f,2.0133333f,1.49f,1.2116667f,10.905f,5.376667f,3.655f,2.6583333f,1.9916667f,1.4716667f,1.1383333f,11.065001f,5.3483334f,3.795f,2.6683335f,1.9850001f,1.42f,1.17f,11.176666f,5.3416667f,3.6633334f,2.665f,1.9816667f,1.5366668f,1.1333333f,11.303333f,5.44f,3.7583332f,2.565f,1.9366667f,1.4483334f,1.1916666f,11.338333f,5.57f,3.6216664f,2.7516668f,1.9816667f,1.4366667f,1.2066667f,11.599999f,5.56f,3.795f,2.6816666f,1.9200001f,1.4916667f,1.2033333f,11.631667f,5.596667f,3.8449998f,2.7166667f,2.0216665f,1.5016667f,1.1366667f,11.861667f,5.7983336f,3.881667f,2.8116665f,1.935f,1.4266667f,1.1533333f,12.030001f,5.733333f,3.8483334f,2.7816665f,2.0516667f,1.475f,1.2433333f,12.138333f,5.94f,3.9316666f,2.77f,1.9383334f,1.5216666f,1.1583333f,12.203334f,5.9483333f,3.9700003f,2.8000002f,2.0816667f,1.535f,1.1766666f};
    final private static int[] dataA = new int[]{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000};

    public static void main(String[] args) throws Exception {
        Plot1 d = new Plot1();
        AnalysisLauncher.open(d);
    }

    @Override
    public void init() {
        // Define a function to plot
        Func3D func = new Func3D((x, y) -> {
            int i, i1;
            double delta;
            if (x >= 1000) {
                i = (int)Math.floor(x / 1000) + 8;
                i1 = i + 1;
                double v = dataErrorTrain[i * 7 + y.intValue()];
                double v1 = dataErrorTrain[i1 * 7 + y.intValue()];
                delta = (v1 - v) * (x / 1000 - Math.floor(x / 1000));
            } else {
                i = (int)(x / 100);
                delta = 0;
            }
            return (double)dataErrorTrain[i * 7 + y.intValue()] + delta;
        });

        final Shape surface = new SurfaceBuilder().orthonormal(new OrthonormalGrid(new Range(0, 18000 - 100), 180, new Range(0, 6), 7), func);
        surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface, new Color(1, 1, 1, .5f)));
        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(true);
        surface.setWireframeColor(Color.BLACK);

        IChartFactory f = new AWTChartFactory();

        chart = f.newChart(Quality.Advanced().setHiDPIEnabled(true));
        chart.getScene().getGraph().add(surface);
    }
}