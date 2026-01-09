async function predict() {
    const data = {
        pl_rade: Number(document.getElementById("pl_rade").value),
        pl_bmasse: Number(document.getElementById("pl_bmasse").value),
        pl_eqt: Number(document.getElementById("pl_eqt").value),
        pl_orbper: Number(document.getElementById("pl_orbper").value),
        pl_orbsmax: Number(document.getElementById("pl_orbsmax").value),
        st_teff: Number(document.getElementById("st_teff").value),
        sy_dist: Number(document.getElementById("sy_dist").value),
    };

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data),
        });

        if (!res.ok) throw new Error("Backend error");

        const result = await res.json();
        let score100 = Math.round(result.habitability_score * 100);

        let label = "";
        if (score100 >= 80) label = "🟢 Highly Habitable";
        else if (score100 >= 60) label = "🟡 Potentially Habitable";
        else if (score100 >= 40) label = "🟠 Marginal";
        else label = "🔴 Not Habitable";

        document.getElementById("result").innerHTML =
            `<h3>${label}</h3><p>Score: ${score100}/100</p>`;

    } catch (err) {
        document.getElementById("result").innerHTML =
            `<p style="color:red;">Error connecting to API</p>`;
    }
}
