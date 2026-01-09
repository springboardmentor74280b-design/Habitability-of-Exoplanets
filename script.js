async function predict() {
    const inputs = document.querySelectorAll("#predict input");
    const data = {};

    inputs.forEach(input => {
        data[input.placeholder] = parseFloat(input.value);
    });

    const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await res.json();
    document.getElementById("prediction-text").innerText =
        result.class_name;

    document.getElementById("result").classList.remove("hidden");
}

async function uploadCSV() {
    const file = document.getElementById("csvFile").files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    document.getElementById("metrics").innerHTML = `
      Accuracy: ${data.accuracy}<br>
      Precision: ${data.precision}<br>
      Recall: ${data.recall}<br>
      F1 Score: ${data.f1}
    `;

    document.getElementById("confusion").src = data.confusion_img;
    document.getElementById("tsne").src = data.tsne_img;
}
