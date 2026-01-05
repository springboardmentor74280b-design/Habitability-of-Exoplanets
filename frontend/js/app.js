const form = document.getElementById("habitabilityForm");
const resultDiv = document.getElementById("result");
const predictionSpan = document.getElementById("prediction");
const scoreSpan = document.getElementById("score");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect inputs
    const data = {
        radius: Number(document.getElementById("radius").value),
        mass: Number(document.getElementById("mass").value),
        temperature: Number(document.getElementById("temperature").value),
        orbital_period: Number(document.getElementById("orbital_period").value)
    };

    try {
        const response = await axios.post(
            "http://127.0.0.1:5000/api/predict",
            data,
            {
                headers: {
                    "Content-Type": "application/json"
                }
            }
        );

        // Extract response
        const label = response.data.habitability_label;
        const probability = response.data.habitability_score;

        // Display result
        predictionSpan.textContent = label;
        scoreSpan.textContent = `${(probability * 100).toFixed(1)}%`;

        resultDiv.classList.remove("d-none");

    } catch (error) {
        alert(
            "Prediction failed: " +
            (error.response?.data?.error || error.message)
        );
    }
});
// -------------------------------
// Load Ranking Table
// -------------------------------
async function loadRankings() {
    try {
        const response = await axios.get("http://127.0.0.1:5000/api/rankings");
        const tableBody = document.getElementById("rankingTable");
        tableBody.innerHTML = "";

        response.data.forEach((planet, index) => {
            const row = `
                <tr>
                    <td>${index + 1}</td>
                    <td>${planet.pl_name || "Unknown"}</td>
                    <td>${planet.pl_rade.toFixed(2)}</td>
                    <td>${planet.pl_eqt.toFixed(1)}</td>
                    <td>${planet.habitability_score.toFixed(4)}</td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
    } catch (error) {
        console.error("Failed to load rankings", error);
    }
}

// Load rankings when page loads
window.onload = loadRankings;
async function loadPlotlyChart() {
    try {
        const response = await axios.get("http://127.0.0.1:5000/api/ranking");
        const data = response.data;

        const radius = data.map(p => p.pl_rade);
        const score = data.map(p => p.habitability_score);
        const names = data.map(p => p.planet_name || "Unknown");

        const trace = {
            x: radius,
            y: score,
            mode: "markers",
            type: "scatter",
            text: names,
            marker: {
                size: 10
            }
        };

        const layout = {
            xaxis: { title: "Planet Radius (Earth Radii)" },
            yaxis: { title: "Habitability Score" },
            margin: { t: 30 }
        };

        Plotly.newPlot("plotlyChart", [trace], layout);
    } catch (error) {
        console.error("Plotly error:", error);
    }
}

// Load chart when page loads
window.addEventListener("load", loadPlotlyChart);
