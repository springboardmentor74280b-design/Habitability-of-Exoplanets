function predict() {
    const resultBox = document.getElementById("result");
    const text = document.getElementById("prediction-text");

    // Dummy prediction logic (frontend only)
    const outcome = Math.random() > 0.5 ? "Habitable 🌍" : "Non-Habitable ❌";

    text.innerText = outcome;
    resultBox.classList.remove("hidden");
}

function resetPrediction() {
    document.getElementById("result").classList.add("hidden");
}

function previewCSV() {
    const fileInput = document.getElementById("csvFile");
    const preview = document.getElementById("csv-preview");

    if (!fileInput.files.length) {
        alert("Please select a CSV file");
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function (e) {
        const rows = e.target.result.split("\n").slice(0, 6);
        preview.innerHTML = "<pre>" + rows.join("\n") + "</pre>";
    };

    reader.readAsText(file);
}
