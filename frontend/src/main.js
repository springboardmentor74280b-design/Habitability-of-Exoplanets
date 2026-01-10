document.getElementById("predictForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  const data = {
    pl_rade: parseFloat(document.getElementById("pl_rade").value),
    pl_bmasse: parseFloat(document.getElementById("pl_bmasse").value),
    pl_orbper: parseFloat(document.getElementById("pl_orbper").value),
    pl_eqt: parseFloat(document.getElementById("pl_eqt").value)
  };

  const response = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  document.getElementById("result").innerText =
    "Prediction Result: " + result.Habitability;
});
