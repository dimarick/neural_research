import org.jzy3d.chart.factories.AWTChartFactory;
import org.jzy3d.chart.factories.IChartFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.plot2d.primitives.LineSerie2d;
import org.jzy3d.plot3d.rendering.legends.overlay.Legend;
import org.jzy3d.plot3d.rendering.legends.overlay.LineLegendLayout;
import org.jzy3d.plot3d.rendering.legends.overlay.OverlayLegendRenderer;
import java.util.ArrayList;

public class Plot2 {
    final private static int[] dataA = new int[]{100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000};
    final private static int[] dataTimeTest = new int[]{327,381,450,406,460,536,686,559,853,933,1198,1537,1961,2592,3077,3245,3647,4234,4731,5301,5657,8446,8475,7507,9749,10709,10624,9530};
    final private static double[] dataErrorTest = new double[]{15.860001,9.36,8.120001,6.72,6.46,6.56,6.79,6.02,5.56,5.59,4.48,4.12,4.77,5.06,3.79,4.28,3.9,3.82,3.66,4.12,3.83,3.26,4.22,4.03,3.88,4.4,3.88,5.42};

    public static void main(String[] args) throws Exception {
        var chart = (new AWTChartFactory()).newChart();
        final var testResult = new LineSerie2d("test errors");
        final var testTime = new LineSerie2d("test time");

        for (var i = 0; i < dataA.length; i++) {
            testResult.add(dataA[i], dataErrorTest[i]);
            testTime.add(dataA[i], dataTimeTest[i]);
        }

        testResult.setWidth(5);
        testTime.setWidth(5);
        testResult.setColor(Color.RED);
        testTime.setColor(Color.GREEN);

        IChartFactory f = new AWTChartFactory();

//        chart.add(testResult);
        chart.add(testTime);

        // Legend
        var infos = new ArrayList<Legend>();
        infos.add(new Legend(testResult.getName(), testResult.getColor()));
        infos.add(new Legend(testTime.getName(), testTime.getColor()));

        OverlayLegendRenderer legend = new OverlayLegendRenderer(infos);
        LineLegendLayout layout = legend.getLayout();

        layout.getMargin().setWidth(10);
        layout.getMargin().setHeight(10);
        layout.setBackgroundColor(Color.WHITE);
        layout.setFont(new java.awt.Font("Helvetica", java.awt.Font.PLAIN, 11));

        chart.addRenderer(legend);

        // Open as 2D chart
        chart.view2d();
        chart.open();
    }
}