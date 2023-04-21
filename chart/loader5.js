var layout = {
    xaxis: {
        autorange: true
    },
        yaxis: {
        type: 'log',
        autorange: true
    }
};

function load(fileName, callback) {
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
      if(xmlhttp.status==200 && xmlhttp.readyState==4){
        callback(xmlhttp.responseText);
      }
    }

    xmlhttp.open('GET', fileName,true);
    xmlhttp.send();
}

function processData(data) {
    var lines = data.split(/\n/);

    var plots = [];
    var currentPlot = -1;
    var currentTrace = 0;
    var currentSize = 0;
    var currentAlgo = null;
    var row = 0;

    for(var i = 0; i < lines.length; i++) {
        var line = lines[i];
        var start = line.match(/Starting test with speed.*\((\d+).0\), volume \d+, Opt: (\w*)/);
        var epoch = line.match(/epoch is (\d+).*Test error rate is: [\d\.]+%. \(([\d\.]+)%\)/);
        if (start !== null) {
            var newSize = parseInt(start[1]);
            if (currentSize !== newSize) {
                currentSize = newSize;
                currentAlgo = null;
                currentPlot++;
                currentTrace = -1;
                plots[currentPlot * 3 + 0] = {
                    traces: [],
                    div: 'chart-' + currentSize.toString() + '-0'
                }
                plots[currentPlot * 3 + 1] = {
                    traces: [],
                    div: 'chart-' + currentSize.toString() + '-1'
                }
                plots[currentPlot * 3 + 2] = {
                    traces: [],
                    div: 'chart-' + currentSize.toString() + '-2'
                }
            }

            if (currentAlgo !== start[2]) {
                currentAlgo = start[2];
                currentTrace++;

                plots[currentPlot * 3 + 0].traces[currentTrace] = {
                    x: [],
                    y: [],
                    n: [],
                    type: 'scatter',
                    name: currentAlgo
                };
                plots[currentPlot * 3 + 1].traces[currentTrace] = {
                    x: [],
                    y: [],
                    n: [],
                    type: 'scatter',
                    name: currentAlgo
                };
                plots[currentPlot * 3 + 2].traces[currentTrace] = {
                    x: [],
                    y: [],
                    n: [],
                    type: 'scatter',
                    name: currentAlgo
                };
            }

            row = 0;
        } else if (epoch !== null) {
            var e = parseInt(epoch[1]);
            var r = parseFloat(epoch[2]);

            plots[currentPlot * 3 + 0].traces[currentTrace].x[row] = e;
            plots[currentPlot * 3 + 1].traces[currentTrace].x[row] = e;
            plots[currentPlot * 3 + 2].traces[currentTrace].x[row] = e;

            var value0 = plots[currentPlot * 3 + 0].traces[currentTrace].y[row];
            var n = plots[currentPlot * 3 + 0].traces[currentTrace].n[row];
            var value1 = plots[currentPlot * 3 + 1].traces[currentTrace].y[row];
            var value2 = plots[currentPlot * 3 + 2].traces[currentTrace].y[row];
            n = n ? n : 0;

            plots[currentPlot * 3 + 0].traces[currentTrace].n[row] = n + 1;

            plots[currentPlot * 3 + 0].traces[currentTrace].y[row] = ((value0 ? value0 : 0) * n + r) / (n +1);
            plots[currentPlot * 3 + 1].traces[currentTrace].y[row] = Math.min((value1 ? value1 : 100), r);
            plots[currentPlot * 3 + 2].traces[currentTrace].y[row] = Math.max((value2 ? value2 : 0), r);

            row++;
        }
    }

    for (var i = 0; i < plots.length; i++) {
        Plotly.newPlot(plots[i].div, plots[i].traces, layout);
    }
}
