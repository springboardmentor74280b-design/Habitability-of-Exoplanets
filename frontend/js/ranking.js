async function loadRankings() {
    const res = await axios.get("http://127.0.0.1:5000/api/rankings");
    const table = document.getElementById("rankingTable");

    res.data.forEach((p, i) => {
        table.innerHTML += `
            <tr>
                <td>${i + 1}</td>
                <td>${p.pl_name || "Unknown"}</td>
                <td>${p.pl_rade.toFixed(2)}</td>
                <td>${p.pl_eqt.toFixed(1)}</td>
                <td>${p.habitability_score.toFixed(4)}</td>
            </tr>
        `;
    });
}
window.onload = loadRankings;
