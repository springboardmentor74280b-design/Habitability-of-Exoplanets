async function loadCharts() {
    const res = await axios.get("http://127.0.0.1:5000/api/rankings");
    const d = res.data;

    // Plot 1: Radius vs Score
    Plotly.newPlot("chart1", [{
        x: d.map(p => p.pl_rade),
        y: d.map(p => p.habitability_score),
        mode: "markers",
        type: "scatter"
    }], { title: "Habitability vs Radius" });

    // Plot 2: Temperature vs Score
    Plotly.newPlot("chart2", [{
        x: d.map(p => p.pl_eqt),
        y: d.map(p => p.habitability_score),
        mode: "markers",
        type: "scatter"
    }], { title: "Habitability vs Temperature" });

    // Plot 3: Radius Distribution
    Plotly.newPlot("chart3", [{
        x: d.map(p => p.pl_rade),
        type: "histogram"
    }], { title: "Planet Radius Distribution" });
}

window.onload = loadCharts;
