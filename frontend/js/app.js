const form = document.getElementById("habitabilityForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
        radius: +radius.value,
        mass: +mass.value,
        temperature: +temperature.value,
        orbital_period: +orbital_period.value
    };

    const res = await axios.post("http://127.0.0.1:5000/api/predict", data);
    prediction.textContent = res.data.habitability_label;
    score.textContent = (res.data.habitability_score * 100).toFixed(2) + "%";
    resultDiv.classList.remove("d-none");
});
