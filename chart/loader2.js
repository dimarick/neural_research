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
        var start = line.match(/Starting test with speed.*\((\d+).0\), volume \d+, (Opt|optimizer): (\w*)/) || line.match(/Starting test with speed.*\((\d+).0\)/);
        var epoch = line.match(/epoch is (\d+) done.*Error rate is: ([\d\.]+)%.*Test error rate is: ([\d\.]+)%. \(([\d\.]+)%\)/) || line.match(/epoch is (\d+) done.*Error rate is: ([\d\.]+)%.*Test error rate is: ([\d\.]+)%/);
        if (start !== null) {
            var newSize = parseInt(start[1]);
            if (currentSize !== newSize) {
                currentSize = newSize;
                currentAlgo = null;
                currentPlot++;
                currentTrace = -1;
                plots[currentPlot] = {
                    traces: [],
                    div: 'chart-' + currentSize.toString()
                }

                plots[currentPlot].traces[0] = {
                    x: [],
                    y: [],
                    n: [],
                    type: 'scatter',
                    name: 'train'
                };

                plots[currentPlot].traces[1] = {
                    x: [],
                    y: [],
                    n: [],
                    type: 'scatter',
                    name: 'test'
                };
            }
            row = 0;
        } else if (epoch !== null) {
            var e = parseInt(epoch[1]);
            var r = parseFloat(epoch[2]);
            var t = parseFloat(epoch[3]);

            plots[currentPlot].traces[0].x[row] = e;
            plots[currentPlot].traces[1].x[row] = e;

            plots[currentPlot].traces[0].y[row] = r;
            plots[currentPlot].traces[1].y[row] = t;

            row++;
        }
    }

    for (var i = 0; i < plots.length; i++) {
        Plotly.newPlot(plots[i].div, plots[i].traces, layout);
    }
}

load('../log/test3.log', processData);