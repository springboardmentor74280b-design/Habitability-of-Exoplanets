// ============================================================================
// EXOPLANET HABITABILITY PREDICTOR - FRONTEND JAVASCRIPT
// API Communication, Form Handling, and Visualizations
// ============================================================================

const API_BASE = '/api';
let probabilityChart = null;
let planetData = null; // Store loaded planet data

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 Exoplanet Habitability Predictor initialized');

    // Load model info
    loadModelInfo();

    // Load planet presets
    loadPlanetPresets();

    // Setup form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handlePrediction);

    // Setup planet selector
    const planetSelector = document.getElementById('planetSelector');
    planetSelector.addEventListener('change', handlePlanetSelection);

    // Add input formatting for all numeric fields (2 decimal places)
    const numericFields = [
        'P_MASS_EST', 'P_RADIUS_EST', 'P_TEMP_EQUIL', 'P_PERIOD', 'P_FLUX',
        'S_MASS', 'S_RADIUS', 'S_TEMP'
    ];

    numericFields.forEach(fieldId => {
        const input = document.getElementById(fieldId);
        if (input) {
            input.addEventListener('blur', function () {
                if (this.value) {
                    const value = parseFloat(this.value);
                    if (!isNaN(value)) {
                        // Format to 2 decimal places
                        this.value = (Math.round(value * 100) / 100).toFixed(2);
                    }
                }
            });
        }
    });

    // Initialize Chart.js defaults
    Chart.defaults.color = '#e2e8f0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
});

// ============================================================================
// API FUNCTIONS
// ============================================================================

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model_info`);
        const data = await response.json();

        // Update F1 score badge
        const f1Score = (data.performance_metrics.f1_score * 100).toFixed(2);
        document.getElementById('modelF1Score').textContent = `${f1Score}% F1`;

        console.log('✓ Model info loaded:', data);
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelF1Score').textContent = '99.65% F1';
    }
}

async function loadExampleData() {
    try {
        const response = await fetch(`${API_BASE}/example`);
        const data = await response.json();

        if (data.example_input) {
            // Populate form with example data (first few fields only for demo)
            const exampleFields = {
                'P_MASS_EST': data.example_input['P_MASS_EST'] || 1.0,
                'P_RADIUS_EST': data.example_input['P_RADIUS_EST'] || 1.0,
                'P_TEMP_EQUIL': data.example_input['P_TEMP_EQUIL'] || 288,
                'P_PERIOD': data.example_input['P_PERIOD'] || 365,
                'P_FLUX': data.example_input['P_FLUX'] || 1.0,
                'S_MASS': data.example_input['S_MASS'] || 1.0,
                'S_RADIUS': data.example_input['S_RADIUS'] || 1.0,
                'S_TEMP': data.example_input['S_TEMP'] || 5778
            };

            Object.keys(exampleFields).forEach(key => {
                const input = document.getElementById(key);
                if (input) {
                    // Format to 2 decimal places
                    const value = parseFloat(exampleFields[key]);
                    input.value = (Math.round(value * 100) / 100).toFixed(2);
                }
            });

            showNotification('Example data loaded! Click "Predict Habitability" to analyze.', 'success');
        }
    } catch (error) {
        console.error('Error loading example:', error);
        showNotification('Could not load example data', 'error');
    }
}

async function loadPlanetPresets() {
    try {
        const response = await fetch(`${API_BASE}/planets`);
        const data = await response.json();

        if (data.error) {
            console.error('Error loading planets:', data.error);
            return;
        }

        planetData = data;
        console.log('✓ Planet presets loaded:', planetData.metadata);

        // Populate dropdown options
        populatePlanetDropdown(data);

    } catch (error) {
        console.error('Error loading planet presets:', error);
    }
}

function populatePlanetDropdown(data) {
    // Populate Earth
    const earthGroup = document.getElementById('earthGroup');
    earthGroup.innerHTML = '';
    const earthOption = document.createElement('option');
    earthOption.value = 'earth';
    earthOption.textContent = `${data.earth.name} - ${data.earth.description}`;
    earthGroup.appendChild(earthOption);

    // Populate Kepler planets
    const keplerGroup = document.getElementById('keplerGroup');
    keplerGroup.innerHTML = '';
    data.kepler.forEach((planet, index) => {
        const option = document.createElement('option');
        option.value = `kepler_${index}`;
        option.textContent = `${planet.name} - ${planet.description}`;
        keplerGroup.appendChild(option);
    });

    // Populate test samples
    const testGroup = document.getElementById('testGroup');
    testGroup.innerHTML = '';
    data.test_samples.forEach((sample, index) => {
        const option = document.createElement('option');
        option.value = `test_${index}`;
        option.textContent = `${sample.name} - ${sample.description}`;
        testGroup.appendChild(option);
    });

    console.log('✓ Dropdown populated with planets');
}

function handlePlanetSelection(event) {
    const selectedValue = event.target.value;
    const infoElement = document.getElementById('selectedPlanetInfo');

    if (!selectedValue || selectedValue === '') {
        // Custom input selected
        infoElement.textContent = '';
        return;
    }

    if (!planetData) {
        showNotification('Planet data not loaded yet', 'warning');
        return;
    }

    let selectedPlanet = null;

    // Parse selection
    if (selectedValue === 'earth') {
        selectedPlanet = planetData.earth;
    } else if (selectedValue.startsWith('kepler_')) {
        const index = parseInt(selectedValue.split('_')[1]);
        selectedPlanet = planetData.kepler[index];
    } else if (selectedValue.startsWith('test_')) {
        const index = parseInt(selectedValue.split('_')[1]);
        selectedPlanet = planetData.test_samples[index];
    }

    if (selectedPlanet) {
        // Populate form fields
        populateFormWithPlanetData(selectedPlanet.data);

        // Update info text
        infoElement.innerHTML = `<strong>Selected:</strong> ${selectedPlanet.name} ${selectedPlanet.description ? '- ' + selectedPlanet.description : ''}`;

        showNotification(`Loaded data for ${selectedPlanet.name}`, 'success');
    }
}

function populateFormWithPlanetData(data) {
    // Populate all form fields with planet data
    Object.keys(data).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            // Format to 2 decimal places
            const value = parseFloat(data[key]);
            input.value = !isNaN(value) ? (Math.round(value * 100) / 100).toFixed(2) : data[key];
            // Add a subtle animation
            input.classList.add('field-updated');
            setTimeout(() => input.classList.remove('field-updated'), 600);
        }
    });
}

async function makePrediction(inputData) {
    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(inputData)
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Prediction failed');
    }

    return await response.json();
}

// ============================================================================
// FORM HANDLING
// ============================================================================

async function handlePrediction(event) {
    event.preventDefault();

    // Show loading state
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = document.getElementById('loadingSpinner');

    btnText.textContent = 'Analyzing...';
    spinner.classList.remove('d-none');
    submitBtn.disabled = true;

    try {
        // Collect form data
        const formData = collectFormData();

        // Make prediction
        const result = await makePrediction(formData);

        // Display results
        displayResults(result);

        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });

        showNotification('Prediction complete!', 'success');

    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(error.message || 'Prediction failed. Please check your inputs.', 'error');
    } finally {
        // Reset button state
        btnText.textContent = 'Predict Habitability';
        spinner.classList.add('d-none');
        submitBtn.disabled = false;
    }
}

function collectFormData() {
    // Collect the 8 simple fields from the form
    const fields = [
        'P_MASS_EST', 'P_RADIUS_EST', 'P_TEMP_EQUIL', 'P_PERIOD', 'P_FLUX',
        'S_MASS', 'S_RADIUS', 'S_TEMP'
    ];

    const data = {};

    fields.forEach(field => {
        const input = document.getElementById(field);
        if (input && input.value) {
            // Round to 2 decimal places
            const value = parseFloat(input.value);
            data[field] = Math.round(value * 100) / 100;
        } else {
            // Provide default values if not filled
            data[field] = 0;
        }
    });

    return data;
}

function clearForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultsSection').classList.add('d-none');
    showNotification('Form cleared', 'info');
}

// ============================================================================
// RESULTS DISPLAY
// ============================================================================

function displayResults(result) {
    const prediction = result.prediction;

    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('d-none');
    resultsSection.classList.add('fade-in');

    // Update prediction class
    const predictionCard = document.getElementById('predictionCard');
    const className = prediction.class_name;
    document.getElementById('predictionClass').textContent = className;

    // Update card color based on prediction (binary classification)
    if (className.includes('Non-Habitable')) {
        predictionCard.style.background = 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)';
    } else {
        predictionCard.style.background = 'linear-gradient(135deg, #10b981 0%, #059669 100%)';
    }

    // Update confidence
    const confidence = (prediction.confidence * 100).toFixed(2);
    document.getElementById('confidenceValue').textContent = `${confidence}%`;

    // Update probability values (binary classification)
    const probs = prediction.probabilities;
    document.getElementById('probNonHab').textContent = `${(probs.non_habitable * 100).toFixed(1)}%`;
    document.getElementById('probHab').textContent = `${(probs.habitable * 100).toFixed(1)}%`;

    // Update chart
    updateProbabilityChart(probs);
}

function updateProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart');

    // Destroy existing chart
    if (probabilityChart) {
        probabilityChart.destroy();
    }

    // Create new chart with binary classification
    probabilityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Non-Habitable', 'Habitable'],
            datasets: [{
                data: [
                    probabilities.non_habitable * 100,
                    probabilities.habitable * 100
                ],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(16, 185, 129, 0.8)'
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(16, 185, 129, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12,
                            family: 'Inter'
                        },
                        color: '#e2e8f0'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${context.parsed.toFixed(2)}%`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================================================
// MODEL INFO MODAL
// ============================================================================

async function showModelInfo() {
    const modal = new bootstrap.Modal(document.getElementById('modelInfoModal'));
    const content = document.getElementById('modelInfoContent');

    modal.show();

    try {
        const response = await fetch(`${API_BASE}/model_info`);
        const data = await response.json();

        const html = `
            <div class="row g-4">
                <div class="col-md-6">
                    <h6 class="fw-bold text-primary mb-3">Model Information</h6>
                    <table class="table table-sm table-dark">
                        <tr><td class="fw-semibold">Name:</td><td>${data.model_info.name}</td></tr>
                        <tr><td class="fw-semibold">Version:</td><td>${data.model_info.version}</td></tr>
                        <tr><td class="fw-semibold">Type:</td><td>${data.model_info.model_type}</td></tr>
                        <tr><td class="fw-semibold">Framework:</td><td>${data.model_info.framework}</td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6 class="fw-bold text-success mb-3">Performance Metrics</h6>
                    <table class="table table-sm table-dark">
                        <tr><td class="fw-semibold">Accuracy:</td><td>${(data.performance_metrics.accuracy * 100).toFixed(2)}%</td></tr>
                        <tr><td class="fw-semibold">Precision:</td><td>${(data.performance_metrics.precision * 100).toFixed(2)}%</td></tr>
                        <tr><td class="fw-semibold">Recall:</td><td>${(data.performance_metrics.recall * 100).toFixed(2)}%</td></tr>
                        <tr><td class="fw-semibold">F1 Score:</td><td class="text-success fw-bold">${(data.performance_metrics.f1_score * 100).toFixed(2)}%</td></tr>
                    </table>
                </div>
                <div class="col-12">
                    <h6 class="fw-bold text-warning mb-3">Training Information</h6>
                    <table class="table table-sm table-dark">
                        <tr><td class="fw-semibold">Training Samples:</td><td>${data.training_info.training_samples.toLocaleString()}</td></tr>
                        <tr><td class="fw-semibold">Test Samples:</td><td>${data.training_info.test_samples.toLocaleString()}</td></tr>
                        <tr><td class="fw-semibold">Features:</td><td>${data.training_info.n_features.toLocaleString()}</td></tr>
                        <tr><td class="fw-semibold">Classes:</td><td>${data.training_info.n_classes}</td></tr>
                    </table>
                </div>
            </div>
        `;

        content.innerHTML = html;

    } catch (error) {
        console.error('Error loading model info:', error);
        content.innerHTML = '<div class="alert alert-danger">Failed to load model information</div>';
    }
}

// ============================================================================
// NOTIFICATIONS
// ============================================================================

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function formatPercentage(num) {
    return `${(num * 100).toFixed(2)}%`;
}
