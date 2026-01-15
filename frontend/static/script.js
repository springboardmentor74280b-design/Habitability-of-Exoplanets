// frontend/static/script.js - COMPLETE REAL-TIME CHART VERSION

// ========== GLOBAL STATE ==========
let currentParameters = {
    radius: 1.0,
    mass: 1.0,
    gravity: 1.0,
    period: 365.0,
    temp: 288.0,
    density: 5.51
};

let currentPrediction = null;
let chartInstances = {
    radar: null,
    distribution: null,
    importance: null
};

// ========== GLOBAL FUNCTIONS FOR HTML ONCLICK ==========

function updateValue(id, value) {
    const valueSpan = document.getElementById(`${id}Value`);
    if (valueSpan) {
        const numValue = parseFloat(value);
        valueSpan.textContent = numValue.toFixed(id === 'temp' ? 0 : 1);
        currentParameters[id] = numValue;
    }
    updateComparisonDisplay();
    updateChartsInRealTime();
}

function updateComparisonDisplay() {
    const radius = parseFloat(document.getElementById('radiusValue').textContent) || 1.0;
    const mass = parseFloat(document.getElementById('massValue').textContent) || 1.0;
    const gravity = parseFloat(document.getElementById('gravityValue').textContent) || 1.0;
    
    document.getElementById('compRadius').textContent = radius.toFixed(1);
    document.getElementById('compMass').textContent = mass.toFixed(1);
    document.getElementById('compGravity').textContent = gravity.toFixed(1);
    
    updatePlanetVisual(radius, mass);
}

function updatePlanetVisual(radius, mass) {
    const planetVisual = document.getElementById('inputPlanetVisual');
    const planetName = document.getElementById('inputPlanetName');
    
    if (!planetVisual || !planetName) return;
    
    const sizeMultiplier = Math.min(radius, 2);
    let planetType = 'Exoplanet';
    let gradient = '';
    let icon = 'fas fa-globe';
    
    if (radius < 0.8) {
        planetType = 'Terrestrial';
        gradient = 'radial-gradient(circle at 30% 30%, #8B4513, #A0522D)';
        icon = 'fas fa-mountain';
    } else if (radius >= 0.8 && radius <= 1.2 && mass >= 0.8 && mass <= 1.2) {
        planetType = 'Earth-like';
        gradient = 'radial-gradient(circle at 30% 30%, #4a90e2, #2d5aa0)';
        icon = 'fas fa-globe-americas';
    } else if (radius > 1.2 && mass > 1.2) {
        planetType = 'Super-Earth';
        gradient = 'radial-gradient(circle at 30% 30%, #FF6347, #FF4500)';
        icon = 'fas fa-expand-arrows-alt';
    } else if (radius > 1.5) {
        planetType = 'Gas Giant';
        gradient = 'radial-gradient(circle at 30% 30%, #FFD700, #FFA500)';
        icon = 'fas fa-gas-pump';
    } else {
        planetType = 'Exoplanet';
        gradient = 'radial-gradient(circle at 30% 30%, #9b59b6, #8e44ad)';
        icon = 'fas fa-star';
    }
    
    planetVisual.style.background = gradient;
    planetVisual.style.transform = `scale(${0.5 + sizeMultiplier * 0.25})`;
    planetVisual.innerHTML = `<i class="${icon}"></i>`;
    planetName.textContent = planetType;
}

function loadSampleData() {
    const sampleSelect = document.getElementById('sampleSelect');
    const selectedSample = sampleSelect ? sampleSelect.value : 'earth';
    
    const samples = {
        'earth': { radius: 1.0, mass: 1.0, gravity: 1.0, period: 365.25, temp: 288, density: 5.51 },
        'super': { radius: 1.5, mass: 5.0, gravity: 2.2, period: 200, temp: 300, density: 6.0 },
        'ocean': { radius: 1.2, mass: 1.5, gravity: 1.1, period: 400, temp: 280, density: 4.0 },
        'mars': { radius: 0.53, mass: 0.11, gravity: 0.38, period: 687, temp: 210, density: 3.93 },
        'hot': { radius: 10.0, mass: 300.0, gravity: 3.0, period: 5, temp: 1500, density: 1.3 }
    };
    
    const sample = samples[selectedSample];
    if (sample) {
        updateSlider('radius', sample.radius);
        updateSlider('mass', sample.mass);
        updateSlider('gravity', sample.gravity);
        updateSlider('period', sample.period);
        updateSlider('temp', sample.temp);
        updateSlider('density', sample.density);
        
        // Update current parameters
        Object.keys(sample).forEach(key => {
            if (currentParameters.hasOwnProperty(key)) {
                currentParameters[key] = sample[key];
            }
        });
        
        document.getElementById('inputPlanetName').textContent = 
            sampleSelect.options[sampleSelect.selectedIndex].text;
    }
}

function updateSlider(sliderId, value) {
    const slider = document.getElementById(sliderId);
    const valueSpan = document.getElementById(`${sliderId}Value`);
    
    if (slider && valueSpan) {
        slider.value = value;
        valueSpan.textContent = value.toFixed(value % 1 === 0 ? 0 : 1);
    }
}

async function predictHabitability() {
    const statusIndicator = document.getElementById('statusIndicator');
    const predictBtn = document.querySelector('.btn-primary');
    
    if (!statusIndicator || !predictBtn) {
        console.error('Required elements not found');
        return;
    }
    
    // Disable button and show loading
    predictBtn.disabled = true;
    predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    setStatusIndicator('Processing...', 'info');
    
    try {
        const inputData = {
            radius: currentParameters.radius,
            mass: currentParameters.mass,
            gravity: currentParameters.gravity,
            period: currentParameters.period,
            temp: currentParameters.temp,
            density: currentParameters.density,
            // Include model-specific parameter names
            P_RADIUS: currentParameters.radius,
            P_MASS: currentParameters.mass,
            P_GRAVITY: currentParameters.gravity,
            P_ORBPER: currentParameters.period,
            P_TEMP_EQUIL: currentParameters.temp,
            P_DENSITY: currentParameters.density
        };
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const result = await response.json();
        
        if (result.success) {
            currentPrediction = result;
            updatePredictionDisplay(result);
            updateAllCharts(result);
            showNotification('Analysis complete!', 'success');
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        setStatusIndicator('Prediction Failed', 'danger');
        showNotification(error.message, 'error');
        showFallbackPrediction();
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="fas fa-bolt"></i> Predict Habitability';
    }
}

function resetForm() {
    // Reset to Earth values
    const earthValues = { radius: 1.0, mass: 1.0, gravity: 1.0, period: 365, temp: 288, density: 5.51 };
    Object.keys(earthValues).forEach(key => {
        updateSlider(key, earthValues[key]);
        updateValue(key, earthValues[key]);
    });
    
    // Reset results
    document.getElementById('scoreValue').textContent = '0';
    document.getElementById('habitabilityLabel').textContent = 'Habitability Score';
    document.getElementById('habitabilityDescription').textContent = 'Enter planetary parameters to get prediction';
    document.getElementById('confidenceValue').textContent = '--';
    
    // Reset probability bars
    ['Non', 'Pot', 'High'].forEach(type => {
        document.getElementById(`prob${type}`).textContent = '0%';
        document.getElementById(`prob${type}Fill`).style.width = '0%';
    });
    
    // Reset status
    setStatusIndicator('Awaiting Input', 'info');
    
    // Reset charts
    resetCharts();
    
    // Reset sample selector
    document.getElementById('sampleSelect').value = '';
    
    showNotification('Form reset to Earth values', 'info');
}

function toggleTheme() {
    const body = document.body;
    const themeToggle = document.querySelector('.theme-toggle i');
    
    if (body.classList.contains('dark-theme')) {
        body.classList.remove('dark-theme');
        themeToggle.className = 'fas fa-sun';
        updateThemeVariables('light');
    } else if (body.classList.contains('light-theme')) {
        body.classList.remove('light-theme');
        themeToggle.className = 'fas fa-moon';
        updateThemeVariables('dark');
    } else {
        body.classList.add('light-theme');
        themeToggle.className = 'fas fa-sun';
        updateThemeVariables('light');
    }
}

function updateThemeVariables(theme) {
    const root = document.documentElement;
    if (theme === 'light') {
        root.style.setProperty('--primary-dark', '#f0f2f5');
        root.style.setProperty('--primary-blue', '#ffffff');
        root.style.setProperty('--text-primary', '#1a1f4b');
        root.style.setProperty('--card-bg', 'rgba(255, 255, 255, 0.95)');
    } else {
        root.style.setProperty('--primary-dark', '#0a0e29');
        root.style.setProperty('--primary-blue', '#1a1f4b');
        root.style.setProperty('--text-primary', '#ffffff');
        root.style.setProperty('--card-bg', 'rgba(26, 31, 75, 0.7)');
    }
}

// ========== CHART FUNCTIONS ==========

function initializeCharts() {
    // Radar Chart
    const radarCtx = document.getElementById('radarChart');
    if (radarCtx) {
        chartInstances.radar = new Chart(radarCtx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Radius', 'Mass', 'Gravity', 'Temperature', 'Orbital Period', 'Density'],
                datasets: [
                    {
                        label: 'Current Planet',
                        data: [50, 50, 50, 50, 50, 50],
                        backgroundColor: 'rgba(74, 144, 226, 0.2)',
                        borderColor: 'rgba(74, 144, 226, 1)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(74, 144, 226, 1)',
                        pointRadius: 4
                    },
                    {
                        label: 'Optimal Range',
                        data: [75, 70, 80, 65, 85, 70],
                        backgroundColor: 'rgba(0, 255, 157, 0.1)',
                        borderColor: 'rgba(0, 255, 157, 0.5)',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { 
                            color: '#b8c1ec',
                            font: { size: 11, family: "'Space Grotesk', sans-serif" }
                        },
                        ticks: { 
                            display: false,
                            backdropColor: 'transparent'
                        },
                        suggestedMin: 0,
                        suggestedMax: 100
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#b8c1ec',
                            font: { size: 12, family: "'Space Grotesk', sans-serif" }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(26, 31, 75, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8c1ec',
                        borderColor: 'rgba(74, 144, 226, 0.5)',
                        borderWidth: 1
                    }
                }
            }
        });
    }
    
    // Distribution Chart
    const distCtx = document.getElementById('distributionChart');
    if (distCtx) {
        chartInstances.distribution = new Chart(distCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Non-Habitable', 'Potentially Habitable', 'Highly Habitable'],
                datasets: [{
                    label: 'Probability (%)',
                    data: [33.3, 33.3, 33.3],
                    backgroundColor: [
                        'rgba(255, 107, 107, 0.8)',
                        'rgba(255, 215, 0, 0.8)',
                        'rgba(0, 255, 157, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 107, 107, 1)',
                        'rgba(255, 215, 0, 1)',
                        'rgba(0, 255, 157, 1)'
                    ],
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: {
                            color: '#b8c1ec',
                            callback: value => value + '%',
                            font: { family: "'Space Grotesk', sans-serif" }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: '#b8c1ec',
                            font: { family: "'Space Grotesk', sans-serif" }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: context => `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`
                        },
                        backgroundColor: 'rgba(26, 31, 75, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8c1ec'
                    }
                }
            }
        });
    }
    
    // Importance Chart
    const impCtx = document.getElementById('importanceChart');
    if (impCtx) {
        chartInstances.importance = new Chart(impCtx.getContext('2d'), {
            type: 'horizontalBar',
            data: {
                labels: ['Temperature', 'Radius', 'Orbital Period', 'Gravity', 'Mass', 'Density'],
                datasets: [{
                    label: 'Impact Score',
                    data: [16.7, 16.7, 16.7, 16.7, 16.7, 16.7],
                    backgroundColor: 'rgba(74, 144, 226, 0.8)',
                    borderColor: 'rgba(74, 144, 226, 1)',
                    borderWidth: 1,
                    borderRadius: 3
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: {
                            color: '#b8c1ec',
                            callback: value => value + '%',
                            font: { family: "'Space Grotesk', sans-serif" }
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: '#b8c1ec',
                            font: { family: "'Space Grotesk', sans-serif" }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: context => `Impact: ${context.parsed.x.toFixed(1)}%`
                        },
                        backgroundColor: 'rgba(26, 31, 75, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#b8c1ec'
                    }
                }
            }
        });
    }
}

async function updateChartsInRealTime() {
    try {
        // Update radar chart
        const radarResponse = await fetch('/api/charts/radar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParameters)
        });
        
        if (radarResponse.ok) {
            const radarData = await radarResponse.json();
            if (radarData.success && chartInstances.radar) {
                updateRadarChart(radarData.radar_data);
            }
        }
        
        // Update feature impact
        const impactResponse = await fetch('/api/charts/feature_impact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParameters)
        });
        
        if (impactResponse.ok) {
            const impactData = await impactResponse.json();
            if (impactData.success && chartInstances.importance) {
                updateImportanceChart(impactData.impact_data);
            }
        }
        
        // Update parameter analysis
        const analysisResponse = await fetch('/api/charts/parameter_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParameters)
        });
        
        if (analysisResponse.ok) {
            const analysisData = await analysisResponse.json();
            if (analysisData.success) {
                updateParameterAnalysis(analysisData.analysis);
            }
        }
        
        // Update class distribution (local calculation)
        updateDistributionChartLocally();
        
    } catch (error) {
        console.log('Real-time update (using local):', error.message);
        updateChartsLocally();
    }
}

function updateAllCharts(predictionResult) {
    if (predictionResult.chart_data) {
        const { radar, feature_impact, parameter_analysis, class_distribution } = predictionResult.chart_data;
        
        if (radar) updateRadarChart(radar);
        if (feature_impact) updateImportanceChart(feature_impact);
        if (parameter_analysis) updateParameterAnalysis(parameter_analysis);
        if (class_distribution) updateDistributionChart(class_distribution);
    }
}

function updateRadarChart(radarData) {
    if (!chartInstances.radar || !radarData) return;
    
    chartInstances.radar.data.datasets[0].data = radarData.values;
    chartInstances.radar.data.datasets[1].data = radarData.optimal;
    chartInstances.radar.update();
    
    // Update radar summary
    if (radarData.summary) {
        const summaryEl = document.querySelector('.analysis-card:nth-child(1) .analysis-header p');
        if (summaryEl) {
            summaryEl.textContent = `Avg deviation: ${radarData.summary.average_deviation}% | Worst: ${radarData.summary.worst_parameter}`;
        }
    }
}

function updateImportanceChart(impactData) {
    if (!chartInstances.importance || !impactData) return;
    
    chartInstances.importance.data.labels = impactData.features;
    chartInstances.importance.data.datasets[0].data = impactData.impacts;
    chartInstances.importance.data.datasets[0].backgroundColor = impactData.colors || 'rgba(74, 144, 226, 0.8)';
    chartInstances.importance.update();
    
    // Update importance summary
    if (impactData.analysis && impactData.analysis.length > 0) {
        const topImpact = impactData.analysis[0];
        const summaryEl = document.querySelector('.analysis-card:nth-child(3) .analysis-header p');
        if (summaryEl) {
            summaryEl.textContent = `Highest impact: ${topImpact.name} (${topImpact.level})`;
        }
    }
}

function updateParameterAnalysis(analysisData) {
    const breakdownGrid = document.getElementById('breakdownGrid');
    if (!breakdownGrid || !analysisData) return;
    
    breakdownGrid.innerHTML = analysisData.map(item => `
        <div class="breakdown-item fade-in">
            <div class="breakdown-header">
                <h4><i class="${item.icon}"></i> ${item.parameter}</h4>
                <span class="breakdown-status" style="color: ${item.status_color || '#b8c1ec'}">
                    ${item.status_text || 'Unknown'}
                </span>
            </div>
            <div class="breakdown-value" style="color: ${item.status_color || '#b8c1ec'}">
                ${item.value}
            </div>
            <div class="breakdown-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${item.percent_of_optimal || 50}%; background: ${item.status_color || '#4a90e2'}"></div>
                </div>
                <span class="progress-label">${item.percent_of_optimal || 50}% of optimal</span>
            </div>
            <div class="breakdown-details">
                <p><strong>Impact:</strong> ${item.impact || 'No impact data'}</p>
                <p><strong>Optimal Range:</strong> ${item.optimal_range || 'Unknown'}</p>
                <p><strong>Recommendation:</strong> ${item.recommendation || 'No recommendation'}</p>
            </div>
        </div>
    `).join('');
}

function updateDistributionChart(distributionData) {
    if (!chartInstances.distribution || !distributionData) return;
    
    const values = distributionData.distribution.map(d => d.value);
    chartInstances.distribution.data.datasets[0].data = values;
    chartInstances.distribution.update();
    
    // Update distribution summary
    if (distributionData.statistics) {
        const stats = distributionData.statistics;
        const summaryEl = document.querySelector('.analysis-card:nth-child(2) .analysis-header p');
        if (summaryEl) {
            summaryEl.textContent = `Dominant: ${stats.dominant_class} (${stats.dominant_probability}%) | Certainty: ${stats.certainty}%`;
        }
    }
}

function updateDistributionChartLocally() {
    if (!chartInstances.distribution) return;
    
    // Calculate simple distribution based on parameters
    const score = calculateHeuristicScoreLocal();
    const probs = [
        Math.max(0, 100 - score - 20), // Non-habitable
        Math.min(40, score * 0.6),     // Potentially habitable
        Math.max(0, score - 40)        // Highly habitable
    ];
    
    // Normalize to 100%
    const total = probs.reduce((a, b) => a + b, 0);
    const normalized = probs.map(p => (p / total) * 100);
    
    chartInstances.distribution.data.datasets[0].data = normalized;
    chartInstances.distribution.update();
}

function updateChartsLocally() {
    // Local calculations for radar chart
    const radarData = {
        values: [
            normalize(currentParameters.radius, 0.1, 5.0),
            normalize(currentParameters.mass, 0.1, 20.0),
            normalize(currentParameters.gravity, 0.1, 3.0),
            normalize(currentParameters.temp, 100, 500),
            normalize(currentParameters.period, 1, 1000),
            normalize(currentParameters.density, 1, 10)
        ],
        optimal: [75, 70, 80, 65, 85, 70],
        summary: {
            average_deviation: 25,
            worst_parameter: 'Temperature',
            best_parameter: 'Radius'
        }
    };
    
    if (chartInstances.radar) {
        chartInstances.radar.data.datasets[0].data = radarData.values;
        chartInstances.radar.update();
    }
    
    // Local calculations for importance chart
    const importanceData = calculateLocalImportance();
    if (chartInstances.importance && importanceData) {
        chartInstances.importance.data.labels = importanceData.features;
        chartInstances.importance.data.datasets[0].data = importanceData.impacts;
        chartInstances.importance.update();
    }
    
    // Local parameter analysis
    updateLocalParameterAnalysis();
    
    // Local distribution
    updateDistributionChartLocally();
}

function normalize(value, min, max) {
    return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}

function calculateLocalImportance() {
    const deviations = {
        'Temperature': Math.abs(currentParameters.temp - 288) / 200 * 100,
        'Radius': Math.abs(currentParameters.radius - 1) * 50,
        'Orbital Period': Math.abs(currentParameters.period - 365) / 500 * 100,
        'Gravity': Math.abs(currentParameters.gravity - 1) * 100,
        'Mass': Math.abs(currentParameters.mass - 1) * 30,
        'Density': Math.abs(currentParameters.density - 5.51) * 20
    };
    
    // Normalize to sum to 100%
    const total = Object.values(deviations).reduce((a, b) => a + b, 0);
    const normalized = Object.keys(deviations).map(key => 
        total > 0 ? (deviations[key] / total) * 100 : 16.7
    );
    
    return {
        features: Object.keys(deviations),
        impacts: normalized.map(v => Math.min(100, v))
    };
}

function updateLocalParameterAnalysis() {
    const analysis = [
        {
            parameter: 'Temperature',
            icon: 'fas fa-thermometer-half',
            value: `${currentParameters.temp}K`,
            status: currentParameters.temp >= 250 && currentParameters.temp <= 320 ? 'optimal' : 
                   currentParameters.temp >= 200 && currentParameters.temp <= 400 ? 'warning' : 'critical',
            status_text: currentParameters.temp >= 250 && currentParameters.temp <= 320 ? 'Optimal' : 
                        currentParameters.temp >= 200 && currentParameters.temp <= 400 ? 'Suboptimal' : 'Critical',
            status_color: currentParameters.temp >= 250 && currentParameters.temp <= 320 ? '#00ff9d' : 
                         currentParameters.temp >= 200 && currentParameters.temp <= 400 ? '#ffd700' : '#ff6b6b',
            percent_of_optimal: Math.min(100, (288 / Math.max(currentParameters.temp, 1)) * 100),
            impact: 'Determines liquid water existence',
            optimal_range: '250-320K',
            recommendation: currentParameters.temp < 250 ? 'Increase temperature' : 
                           currentParameters.temp > 320 ? 'Decrease temperature' : 'Optimal range'
        },
        {
            parameter: 'Radius',
            icon: 'fas fa-expand-alt',
            value: `${currentParameters.radius}R⊕`,
            status: currentParameters.radius >= 0.8 && currentParameters.radius <= 1.5 ? 'optimal' : 
                   currentParameters.radius >= 0.5 && currentParameters.radius <= 2.5 ? 'warning' : 'critical',
            status_text: currentParameters.radius >= 0.8 && currentParameters.radius <= 1.5 ? 'Optimal' : 
                        currentParameters.radius >= 0.5 && currentParameters.radius <= 2.5 ? 'Suboptimal' : 'Critical',
            status_color: currentParameters.radius >= 0.8 && currentParameters.radius <= 1.5 ? '#00ff9d' : 
                         currentParameters.radius >= 0.5 && currentParameters.radius <= 2.5 ? '#ffd700' : '#ff6b6b',
            percent_of_optimal: Math.min(100, (1 / Math.max(currentParameters.radius, 0.1)) * 100),
            impact: 'Affects gravity and atmosphere',
            optimal_range: '0.8-1.5R⊕',
            recommendation: currentParameters.radius < 0.8 ? 'Larger size needed' : 
                           currentParameters.radius > 1.5 ? 'Smaller size beneficial' : 'Optimal range'
        },
        {
            parameter: 'Orbital Period',
            icon: 'fas fa-sync-alt',
            value: `${currentParameters.period} days`,
            status: currentParameters.period >= 200 && currentParameters.period <= 400 ? 'optimal' : 
                   currentParameters.period >= 100 && currentParameters.period <= 600 ? 'warning' : 'critical',
            status_text: currentParameters.period >= 200 && currentParameters.period <= 400 ? 'Optimal' : 
                        currentParameters.period >= 100 && currentParameters.period <= 600 ? 'Suboptimal' : 'Critical',
            status_color: currentParameters.period >= 200 && currentParameters.period <= 400 ? '#00ff9d' : 
                         currentParameters.period >= 100 && currentParameters.period <= 600 ? '#ffd700' : '#ff6b6b',
            percent_of_optimal: Math.min(100, (365 / Math.max(currentParameters.period, 1)) * 100),
            impact: 'Affects climate stability',
            optimal_range: '200-400 days',
            recommendation: currentParameters.period < 200 ? 'Longer period needed' : 
                           currentParameters.period > 400 ? 'Shorter period beneficial' : 'Optimal range'
        },
        {
            parameter: 'Gravity',
            icon: 'fas fa-weight',
            value: `${currentParameters.gravity}g`,
            status: currentParameters.gravity >= 0.8 && currentParameters.gravity <= 1.2 ? 'optimal' : 
                   currentParameters.gravity >= 0.5 && currentParameters.gravity <= 1.8 ? 'warning' : 'critical',
            status_text: currentParameters.gravity >= 0.8 && currentParameters.gravity <= 1.2 ? 'Optimal' : 
                        currentParameters.gravity >= 0.5 && currentParameters.gravity <= 1.8 ? 'Suboptimal' : 'Critical',
            status_color: currentParameters.gravity >= 0.8 && currentParameters.gravity <= 1.2 ? '#00ff9d' : 
                         currentParameters.gravity >= 0.5 && currentParameters.gravity <= 1.8 ? '#ffd700' : '#ff6b6b',
            percent_of_optimal: Math.min(100, (1 / Math.max(currentParameters.gravity, 0.1)) * 100),
            impact: 'Affects atmosphere retention',
            optimal_range: '0.8-1.2g',
            recommendation: currentParameters.gravity < 0.8 ? 'Higher gravity needed' : 
                           currentParameters.gravity > 1.2 ? 'Lower gravity beneficial' : 'Optimal range'
        }
    ];
    
    updateParameterAnalysis(analysis);
}

function resetCharts() {
    // Reset radar chart
    if (chartInstances.radar) {
        chartInstances.radar.data.datasets[0].data = [50, 50, 50, 50, 50, 50];
        chartInstances.radar.update();
    }
    
    // Reset distribution chart
    if (chartInstances.distribution) {
        chartInstances.distribution.data.datasets[0].data = [33.3, 33.3, 33.3];
        chartInstances.distribution.update();
    }
    
    // Reset importance chart
    if (chartInstances.importance) {
        chartInstances.importance.data.datasets[0].data = [16.7, 16.7, 16.7, 16.7, 16.7, 16.7];
        chartInstances.importance.update();
    }
    
    // Reset analysis breakdown
    const breakdownGrid = document.getElementById('breakdownGrid');
    if (breakdownGrid) {
        breakdownGrid.innerHTML = `
            <div class="breakdown-placeholder">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Adjust parameters to see detailed analysis</p>
            </div>
        `;
    }
}

// ========== PREDICTION DISPLAY FUNCTIONS ==========

function updatePredictionDisplay(result) {
    // Update score wheel
    updateScoreWheel(result.probability);
    
    // Update status
    setStatusIndicator(result.prediction_label, 
        result.prediction === 0 ? 'danger' : 
        result.prediction === 1 ? 'warning' : 'success');
    
    // Update labels
    document.getElementById('habitabilityLabel').textContent = result.prediction_label;
    document.getElementById('habitabilityDescription').textContent = 
        getHabitabilityDescription(result.prediction, result.confidence);
    document.getElementById('confidenceValue').textContent = result.confidence;
    document.getElementById('modelUsed').textContent = result.model_used;
    document.getElementById('earthSimilarity').textContent = `${result.earth_similarity || '50'}%`;
    
    // Update probability bars
    updateProbabilityBars(result.probabilities);
}

function updateScoreWheel(score) {
    document.getElementById('scoreValue').textContent = score.toFixed(1);
    if (window.drawScoreWheel) {
        window.drawScoreWheel(score);
    }
}

function updateProbabilityBars(probabilities) {
    const types = ['Non', 'Pot', 'High'];
    const keys = ['Non_Habitable', 'Potentially_Habitable', 'Highly_Habitable'];
    
    types.forEach((type, index) => {
        const value = probabilities[keys[index]];
        const valueEl = document.getElementById(`prob${type}`);
        const fillEl = document.getElementById(`prob${type}Fill`);
        
        if (valueEl && fillEl) {
            const numValue = parseFloat(value) || 0;
            valueEl.textContent = `${numValue.toFixed(1)}%`;
            fillEl.style.width = `${numValue}%`;
        }
    });
}

function getHabitabilityDescription(prediction, confidence) {
    const descriptions = [
        `Unlikely to support Earth-like life (${confidence} confidence). Extreme conditions prevent stable biosphere formation.`,
        `Promising candidate for habitability (${confidence} confidence). Shows characteristics suitable for potential life forms.`,
        `Excellent candidate for Earth-like life (${confidence} confidence). Strong indicators of habitable conditions.`
    ];
    return descriptions[prediction] || 'Analysis complete.';
}

function setStatusIndicator(text, type) {
    const indicator = document.getElementById('statusIndicator');
    if (!indicator) return;
    
    const icons = {
        'success': 'fas fa-check-circle',
        'warning': 'fas fa-exclamation-triangle',
        'danger': 'fas fa-times-circle',
        'info': 'fas fa-info-circle'
    };
    
    indicator.innerHTML = `<i class="${icons[type] || icons.info}"></i> ${text}`;
    indicator.className = `status-indicator status-${type}`;
}

// ========== FALLBACK FUNCTIONS ==========

function showFallbackPrediction() {
    const score = calculateHeuristicScoreLocal();
    const prediction = score < 30 ? 0 : score < 70 ? 1 : 2;
    
    const result = {
        success: true,
        prediction: prediction,
        prediction_label: ['Non-Habitable', 'Potentially Habitable', 'Highly Habitable'][prediction],
        probability: score,
        probabilities: {
            Non_Habitable: (100 - score) * 0.4,
            Potentially_Habitable: (100 - score) * 0.3,
            Highly_Habitable: score
        },
        earth_similarity: calculateEarthSimilarityLocal(),
        model_used: 'Heuristic Analysis',
        confidence: getConfidenceLevel(score),
        fallback: true
    };
    
    updatePredictionDisplay(result);
    updateChartsLocally();
    showNotification('Using heuristic analysis (model offline)', 'warning');
}

function calculateHeuristicScoreLocal() {
    let score = 50;
    
    // Radius contribution
    if (currentParameters.radius >= 0.8 && currentParameters.radius <= 1.5) score += 20;
    else if (currentParameters.radius >= 0.5 && currentParameters.radius <= 2.5) score += 5;
    else score -= 20;
    
    // Temperature contribution
    if (currentParameters.temp >= 250 && currentParameters.temp <= 320) score += 25;
    else if (currentParameters.temp >= 200 && currentParameters.temp <= 400) score += 5;
    else score -= 25;
    
    // Gravity contribution
    if (currentParameters.gravity >= 0.8 && currentParameters.gravity <= 1.2) score += 20;
    else if (currentParameters.gravity >= 0.5 && currentParameters.gravity <= 2.0) score += 5;
    else score -= 15;
    
    return Math.max(0, Math.min(100, score));
}

function calculateEarthSimilarityLocal() {
    let similarity = 0;
    
    if (Math.abs(currentParameters.radius - 1) < 0.2) similarity += 25;
    if (Math.abs(currentParameters.mass - 1) < 0.3) similarity += 20;
    if (Math.abs(currentParameters.gravity - 1) < 0.2) similarity += 20;
    if (Math.abs(currentParameters.temp - 288) < 30) similarity += 20;
    if (Math.abs(currentParameters.density - 5.51) < 1) similarity += 15;
    
    return Math.min(100, similarity);
}

function getConfidenceLevel(probability) {
    if (probability >= 90) return 'Very High';
    if (probability >= 75) return 'High';
    if (probability >= 60) return 'Moderate';
    if (probability >= 40) return 'Low';
    return 'Very Low';
}

// ========== SCORE WHEEL FUNCTIONS ==========

function initializeScoreWheel() {
    const canvas = document.getElementById('scoreCanvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    window.scoreCtx = ctx;
    
    // Make drawScoreWheel available globally
    window.drawScoreWheel = function(score) {
        if (!ctx) return;
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 80;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw background circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(74, 144, 226, 0.2)';
        ctx.lineWidth = 12;
        ctx.stroke();
        
        // Draw progress arc
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (score / 100) * 2 * Math.PI;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, startAngle, endAngle);
        
        // Gradient based on score
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        if (score < 33) {
            gradient.addColorStop(0, '#ff6b6b');
            gradient.addColorStop(1, '#ffb347');
        } else if (score < 66) {
            gradient.addColorStop(0, '#ffb347');
            gradient.addColorStop(1, '#00d4aa');
        } else {
            gradient.addColorStop(0, '#00d4aa');
            gradient.addColorStop(1, '#4a90e2');
        }
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 12;
        ctx.lineCap = 'round';
        ctx.stroke();
    };
    
    // Draw initial wheel
    window.drawScoreWheel(0);
}

// ========== NOTIFICATION SYSTEM ==========

function showNotification(message, type = 'info') {
    // Remove existing notifications
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)}"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        max-width: 400px;
        font-family: 'Space Grotesk', sans-serif;
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOut 0.0s ease-in';
            setTimeout(() => notification.remove(), 5);
        }
    }, 5000);
    
    // Add animation styles if not already present
    if (!document.getElementById('notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

function getNotificationIcon(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

function getNotificationColor(type) {
    switch(type) {
        case 'success': return 'rgba(0, 255, 157, 0.9)';
        case 'error': return 'rgba(255, 107, 107, 0.9)';
        case 'warning': return 'rgba(255, 215, 0, 0.9)';
        default: return 'rgba(74, 144, 226, 0.9)';
    }
}

// ========== VISUALIZATION FUNCTIONS ==========

async function loadVisualizations() {
    const grid = document.getElementById('visualizationsGrid');
    if (!grid) return;
    
    try {
        const response = await fetch('/api/visualizations');
        const data = await response.json();
        
        grid.innerHTML = '';
        data.visualizations.forEach((viz, index) => {
            const card = document.createElement('div');
            card.className = 'visualization-card fade-in';
            card.style.animationDelay = `${index * 0.1}s`;
            card.innerHTML = `
                <div class="viz-image-container">
                    <img src="${viz.path}" alt="${viz.name}" class="viz-image" 
                         onerror="this.src='/static/image.jpg'">
                    <div class="viz-overlay">
                        <i class="fas fa-search-plus"></i>
                    </div>
                </div>
                <div class="viz-content">
                    <h4>${viz.name}</h4>
                    <p>${viz.description}</p>
                </div>
            `;
            
            // Add click to enlarge
            card.addEventListener('click', function() {
                showImageModal(viz.path, viz.name, viz.description);
            });
            
            grid.appendChild(card);
        });
    } catch (error) {
        console.error('Error loading visualizations:', error);
        showDefaultVisualizations();
    }
}

function showImageModal(src, title, description) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('imageModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'imageModal';
        modal.className = 'image-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <span class="close-modal" onclick="closeImageModal()">&times;</span>
                <h3 id="modalTitle"></h3>
                <img id="modalImage" src="" alt="">
                <p id="modalDescription"></p>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close modal on outside click
        modal.addEventListener('click', function(e) {
            if (e.target === modal) {
                closeImageModal();
            }
        });
    }
    
    // Set modal content
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalImage').src = src;
    document.getElementById('modalDescription').textContent = description;
    
    // Show modal
    modal.style.display = 'flex';
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function showDefaultVisualizations() {
    const grid = document.getElementById('visualizationsGrid');
    if (!grid) return;
    
    const defaultViz = [
        {
            name: 'Feature Analysis',
            path: '/static/image.jpg',
            description: 'Comprehensive analysis of planetary features'
        },
        {
            name: 'Model Performance',
            path: '/static/image.jpg',
            description: 'Evaluation of machine learning models'
        },
        {
            name: 'Habitability Distribution',
            path: '/static/image.jpg',
            description: 'Distribution of habitable exoplanets'
        }
    ];
    
    grid.innerHTML = '';
    defaultViz.forEach((viz, index) => {
        const card = document.createElement('div');
        card.className = 'visualization-card fade-in';
        card.style.animationDelay = `${index * 0.1}s`;
        card.innerHTML = `
            <div class="viz-image-container">
                <img src="${viz.path}" alt="${viz.name}" class="viz-image">
                <div class="viz-overlay">
                    <i class="fas fa-search-plus"></i>
                </div>
            </div>
            <div class="viz-content">
                <h4>${viz.name}</h4>
                <p>${viz.description}</p>
            </div>
        `;
        
        card.addEventListener('click', function() {
            showImageModal(viz.path, viz.name, viz.description);
        });
        
        grid.appendChild(card);
    });
}

// ========== MOBILE MENU FUNCTION ==========

function initMobileMenu() {
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuBtn && navLinks) {
        // Only show mobile menu button on small screens
        if (window.innerWidth <= 768) {
            mobileMenuBtn.style.display = 'flex';
            mobileMenuBtn.innerHTML = '<i class="fas fa-bars"></i>';
            
            mobileMenuBtn.addEventListener('click', function() {
                navLinks.classList.toggle('show');
                mobileMenuBtn.innerHTML = navLinks.classList.contains('show') 
                    ? '<i class="fas fa-times"></i>' 
                    : '<i class="fas fa-bars"></i>';
            });
            
            // Close menu when clicking outside
            document.addEventListener('click', function(event) {
                if (!navLinks.contains(event.target) && !mobileMenuBtn.contains(event.target)) {
                    navLinks.classList.remove('show');
                    mobileMenuBtn.innerHTML = '<i class="fas fa-bars"></i>';
                }
            });
            
            // Close menu on link click
            navLinks.querySelectorAll('a').forEach(link => {
                link.addEventListener('click', function() {
                    navLinks.classList.remove('show');
                    mobileMenuBtn.innerHTML = '<i class="fas fa-bars"></i>';
                });
            });
        } else {
            mobileMenuBtn.style.display = 'none';
        }
    }
}

// ========== ANIMATION FUNCTIONS ==========

function animateScoreChange(targetScore) {
    if (window.animationFrame) {
        cancelAnimationFrame(window.animationFrame);
    }
    
    const startScore = window.currentScore || 0;
    const duration = 1500;
    const startTime = performance.now();
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        
        window.currentScore = startScore + (targetScore - startScore) * easeProgress;
        if (window.drawScoreWheel) {
            window.drawScoreWheel(window.currentScore);
        }
        document.getElementById('scoreValue').textContent = Math.round(window.currentScore);
        
        if (progress < 1) {
            window.animationFrame = requestAnimationFrame(animate);
        } else {
            window.currentScore = targetScore;
        }
    }
    
    window.animationFrame = requestAnimationFrame(animate);
}

// ========== INITIALIZATION ==========

document.addEventListener('DOMContentLoaded', function() {
    // Initialize score wheel
    initializeScoreWheel();
    
    // Initialize charts
    initializeCharts();
    
    // Initialize with Earth values
    updateComparisonDisplay();
    updatePlanetVisual(1.0, 1.0);
    
    // Set up real-time updates
    setupRealTimeUpdates();
    
    // Load visualizations
    loadVisualizations();
    
    // Set up smooth scrolling
    setupSmoothScrolling();
    
    // Set up keyboard shortcuts
    setupKeyboardShortcuts();
    
    // Initialize mobile menu
    initMobileMenu();
    
    // Update mobile menu on resize
    window.addEventListener('resize', initMobileMenu);
    
    // Load Earth sample by default
    setTimeout(() => {
        const select = document.getElementById('sampleSelect');
        if (select) {
            select.value = 'earth';
            loadSampleData();
        }
    }, 500);
    
    console.log('🚀 AstroHab initialized with real-time charts');
});

function setupRealTimeUpdates() {
    // Debounced chart updates on slider change
    let updateTimeout;
    document.querySelectorAll('.input-slider').forEach(slider => {
        slider.addEventListener('input', function() {
            clearTimeout(updateTimeout);
            updateTimeout = setTimeout(updateChartsInRealTime, 300);
        });
    });
}

function setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const target = document.querySelector(targetId);
            if (target) {
                window.scrollTo({
                    top: target.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl+Enter to predict
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            predictHabitability();
        }
        // Escape to reset
        if (e.key === 'Escape') {
            resetForm();
        }
        // Number keys for samples (1-5)
        if (e.key >= '1' && e.key <= '5') {
            const samples = ['earth', 'super', 'ocean', 'mars', 'hot'];
            const index = parseInt(e.key) - 1;
            if (samples[index]) {
                document.getElementById('sampleSelect').value = samples[index];
                loadSampleData();
            }
        }
    });
}

// Make global functions available
window.updateValue = updateValue;
window.loadSampleData = loadSampleData;
window.predictHabitability = predictHabitability;
window.resetForm = resetForm;
window.toggleTheme = toggleTheme;
window.closeImageModal = closeImageModal;