// Mock data for Exoplanet Habitability Predictor

// Known exoplanets database
const EXOPLANET_DATABASE = [
    {
        id: 'proxima-centauri-b',
        name: 'Proxima Centauri b',
        discoveryYear: 2016,
        starType: 'M-dwarf',
        starName: 'Proxima Centauri',
        distance: '4.24 ly',
        
        // Required parameters
        planetRadius: 1.07, // Earth radii
        planetMass: 1.27, // Earth masses
        planetTemp: 234, // Kelvin (equilibrium)
        orbitalPeriod: 11.2, // days
        starRadius: 0.14, // Solar radii
        starTemp: 3042, // Kelvin
        
        // Advanced parameters
        stellarLuminosity: 0.0015, // Solar luminosities
        atmosphericComposition: 'unknown',
        waterPresence: 'maybe',
        magneticField: 0.8,
        
        // Habitability results
        habitabilityScore: 87,
        habitabilityClass: 'Potentially Habitable',
        classCategory: 'Earth-like',
        
        // Secondary metrics
        tempSuitability: 92,
        atmosphericLikelihood: 78,
        waterProbability: 85,
        radiationTolerance: 76,
        
        // Additional info
        description: 'Closest known exoplanet to Earth, orbiting within habitable zone of Proxima Centauri.',
        interestingFact: 'Receives about 65% of the irradiation Earth gets from the Sun.',
        imageColor: '#4CAF50'
    },
    {
        id: 'trappist-1e',
        name: 'TRAPPIST-1e',
        discoveryYear: 2017,
        starType: 'Ultra-cool dwarf',
        starName: 'TRAPPIST-1',
        distance: '39.5 ly',
        
        planetRadius: 0.92,
        planetMass: 0.69,
        planetTemp: 251,
        orbitalPeriod: 6.1,
        starRadius: 0.12,
        starTemp: 2566,
        
        stellarLuminosity: 0.0005,
        atmosphericComposition: 'nitrogen-oxygen',
        waterPresence: 'yes',
        magneticField: 0.6,
        
        habitabilityScore: 91,
        habitabilityClass: 'Highly Habitable',
        classCategory: 'Super-Earth',
        
        tempSuitability: 88,
        atmosphericLikelihood: 82,
        waterProbability: 92,
        radiationTolerance: 71,
        
        description: 'One of seven Earth-sized planets in TRAPPIST-1 system, with high potential for liquid water.',
        interestingFact: 'Part of the most studied planetary system outside our own.',
        imageColor: '#2196F3'
    },
    {
        id: 'kepler-452b',
        name: 'Kepler-452b',
        discoveryYear: 2015,
        starType: 'G-type',
        starName: 'Kepler-452',
        distance: '1400 ly',
        
        planetRadius: 1.63,
        planetMass: 5.0,
        planetTemp: 265,
        orbitalPeriod: 385,
        starRadius: 1.11,
        starTemp: 5757,
        
        stellarLuminosity: 1.2,
        atmosphericComposition: 'carbon-dioxide',
        waterPresence: 'maybe',
        magneticField: 1.2,
        
        habitabilityScore: 83,
        habitabilityClass: 'Potentially Habitable',
        classCategory: 'Super-Earth',
        
        tempSuitability: 85,
        atmosphericLikelihood: 75,
        waterProbability: 80,
        radiationTolerance: 82,
        
        description: 'First near-Earth-size world discovered in habitable zone of a Sun-like star.',
        interestingFact: 'Often called "Earth\'s cousin" due to similar characteristics.',
        imageColor: '#FF9800'
    },
    {
        id: 'kepler-186f',
        name: 'Kepler-186f',
        discoveryYear: 2014,
        starType: 'M-dwarf',
        starName: 'Kepler-186',
        distance: '582 ly',
        
        planetRadius: 1.17,
        planetMass: 1.4,
        planetTemp: 188,
        orbitalPeriod: 130,
        starRadius: 0.47,
        starTemp: 3755,
        
        stellarLuminosity: 0.04,
        atmosphericComposition: 'unknown',
        waterPresence: 'maybe',
        magneticField: 0.9,
        
        habitabilityScore: 79,
        habitabilityClass: 'Marginally Habitable',
        classCategory: 'Earth-like',
        
        tempSuitability: 72,
        atmosphericLikelihood: 68,
        waterProbability: 65,
        radiationTolerance: 75,
        
        description: 'First Earth-size planet discovered in habitable zone of another star.',
        interestingFact: 'Receives about one-third the energy from its star that Earth gets from the Sun.',
        imageColor: '#9C27B0'
    },
    {
        id: 'gj-667cc',
        name: 'GJ 667Cc',
        discoveryYear: 2011,
        starType: 'M-dwarf',
        starName: 'GJ 667C',
        distance: '23.6 ly',
        
        planetRadius: 1.54,
        planetMass: 3.8,
        planetTemp: 277,
        orbitalPeriod: 28.1,
        starRadius: 0.42,
        starTemp: 3700,
        
        stellarLuminosity: 0.013,
        atmosphericComposition: 'nitrogen-oxygen',
        waterPresence: 'yes',
        magneticField: 1.1,
        
        habitabilityScore: 88,
        habitabilityClass: 'Highly Habitable',
        classCategory: 'Super-Earth',
        
        tempSuitability: 90,
        atmosphericLikelihood: 80,
        waterProbability: 88,
        radiationTolerance: 74,
        
        description: 'Super-Earth orbiting within habitable zone of a nearby red dwarf.',
        interestingFact: 'Receives about 90% of the light Earth receives from the Sun.',
        imageColor: '#00BCD4'
    },
    {
        id: 'hd-40307g',
        name: 'HD 40307g',
        discoveryYear: 2012,
        starType: 'K-type',
        starName: 'HD 40307',
        distance: '42.4 ly',
        
        planetRadius: 2.39,
        planetMass: 7.1,
        planetTemp: 279,
        orbitalPeriod: 197.8,
        starRadius: 0.72,
        starTemp: 4977,
        
        stellarLuminosity: 0.23,
        atmosphericComposition: 'hydrogen-helium',
        waterPresence: 'maybe',
        magneticField: 1.5,
        
        habitabilityScore: 65,
        habitabilityClass: 'Possibly Habitable',
        classCategory: 'Super-Earth',
        
        tempSuitability: 78,
        atmosphericLikelihood: 55,
        waterProbability: 60,
        radiationTolerance: 85,
        
        description: 'Super-Earth in the habitable zone of an orange dwarf star.',
        interestingFact: 'May have a thick atmosphere and could be a "water world".',
        imageColor: '#E91E63'
    },
    {
        id: 'kepler-22b',
        name: 'Kepler-22b',
        discoveryYear: 2011,
        starType: 'G-type',
        starName: 'Kepler-22',
        distance: '635 ly',
        
        planetRadius: 2.38,
        planetMass: 6.4,
        planetTemp: 262,
        orbitalPeriod: 290,
        starRadius: 0.98,
        starTemp: 5518,
        
        stellarLuminosity: 0.79,
        atmosphericComposition: 'unknown',
        waterPresence: 'maybe',
        magneticField: 1.3,
        
        habitabilityScore: 72,
        habitabilityClass: 'Potentially Habitable',
        classCategory: 'Super-Earth',
        
        tempSuitability: 80,
        atmosphericLikelihood: 62,
        waterProbability: 70,
        radiationTolerance: 78,
        
        description: 'First Kepler planet found in habitable zone of a Sun-like star.',
        interestingFact: 'If it has an Earth-like atmosphere, surface temperature would be about 22°C.',
        imageColor: '#FF5722'
    },
    {
        id: 'tau-ceti-e',
        name: 'Tau Ceti e',
        discoveryYear: 2012,
        starType: 'G-type',
        starName: 'Tau Ceti',
        distance: '11.9 ly',
        
        planetRadius: 1.59,
        planetMass: 3.9,
        planetTemp: 311,
        orbitalPeriod: 162.9,
        starRadius: 0.79,
        starTemp: 5344,
        
        stellarLuminosity: 0.52,
        atmosphericComposition: 'carbon-dioxide',
        waterPresence: 'no',
        magneticField: 1.0,
        
        habitabilityScore: 58,
        habitabilityClass: 'Marginally Habitable',
        classCategory: 'Super-Earth',
        
        tempSuitability: 65,
        atmosphericLikelihood: 58,
        waterProbability: 45,
        radiationTolerance: 70,
        
        description: 'Super-Earth orbiting one of the closest Sun-like stars.',
        interestingFact: 'May be too hot for liquid water unless it has strong reflective clouds.',
        imageColor: '#795548'
    }
];

// Example configurations for quick loading
const EXAMPLE_CONFIGURATIONS = {
    earth: {
        planetRadius: 1.0,
        planetMass: 1.0,
        planetTemp: 288,
        orbitalPeriod: 365,
        starRadius: 1.0,
        starTemp: 5778,
        stellarLuminosity: 1.0,
        atmosphericComposition: 'nitrogen-oxygen',
        waterPresence: 'yes',
        magneticField: 1.0
    },
    proxima: {
        planetRadius: 1.07,
        planetMass: 1.27,
        planetTemp: 234,
        orbitalPeriod: 11.2,
        starRadius: 0.14,
        starTemp: 3042,
        stellarLuminosity: 0.0015,
        atmosphericComposition: 'unknown',
        waterPresence: 'maybe',
        magneticField: 0.8
    },
    trappist: {
        planetRadius: 0.92,
        planetMass: 0.69,
        planetTemp: 251,
        orbitalPeriod: 6.1,
        starRadius: 0.12,
        starTemp: 2566,
        stellarLuminosity: 0.0005,
        atmosphericComposition: 'nitrogen-oxygen',
        waterPresence: 'yes',
        magneticField: 0.6
    },
    kepler: {
        planetRadius: 1.63,
        planetMass: 5.0,
        planetTemp: 265,
        orbitalPeriod: 385,
        starRadius: 1.11,
        starTemp: 5757,
        stellarLuminosity: 1.2,
        atmosphericComposition: 'carbon-dioxide',
        waterPresence: 'maybe',
        magneticField: 1.2
    }
};

// Habitability classification system
const HABITABILITY_CLASSES = {
    'Highly Habitable': {
        minScore: 85,
        description: 'Excellent conditions for Earth-like life',
        color: '#4CAF50',
        icon: 'fa-star'
    },
    'Potentially Habitable': {
        minScore: 70,
        description: 'Good potential for supporting life',
        color: '#2196F3',
        icon: 'fa-globe-americas'
    },
    'Marginally Habitable': {
        minScore: 50,
        description: 'Limited potential for habitability',
        color: '#FF9800',
        icon: 'fa-exclamation-triangle'
    },
    'Possibly Habitable': {
        minScore: 30,
        description: 'Some potential under specific conditions',
        color: '#9C27B0',
        icon: 'fa-question-circle'
    },
    'Unlikely Habitable': {
        minScore: 0,
        description: 'Poor conditions for known life forms',
        color: '#F44336',
        icon: 'fa-times-circle'
    }
};

// Star classification system
const STAR_TYPES = {
    'O': {
        tempRange: [30000, 50000],
        color: 'blue',
        size: 'very large',
        luminosity: 'very high',
        lifespan: 'short'
    },
    'B': {
        tempRange: [10000, 30000],
        color: 'blue-white',
        size: 'large',
        luminosity: 'high',
        lifespan: 'short'
    },
    'A': {
        tempRange: [7500, 10000],
        color: 'white',
        size: 'medium-large',
        luminosity: 'medium-high',
        lifespan: 'medium'
    },
    'F': {
        tempRange: [6000, 7500],
        color: 'yellow-white',
        size: 'medium',
        luminosity: 'medium',
        lifespan: 'medium-long'
    },
    'G': {
        tempRange: [5200, 6000],
        color: 'yellow',
        size: 'medium',
        luminosity: 'medium',
        lifespan: 'long',
        example: 'Sun'
    },
    'K': {
        tempRange: [3700, 5200],
        color: 'orange',
        size: 'medium-small',
        luminosity: 'medium-low',
        lifespan: 'very long'
    },
    'M': {
        tempRange: [2400, 3700],
        color: 'red',
        size: 'small',
        luminosity: 'low',
        lifespan: 'extremely long'
    }
};

// Atmospheric compositions
const ATMOSPHERIC_COMPOSITIONS = {
    'nitrogen-oxygen': {
        name: 'Nitrogen-Oxygen',
        earthSimilarity: 1.0,
        description: 'Earth-like atmosphere with 78% N₂, 21% O₂',
        color: '#4CAF50'
    },
    'carbon-dioxide': {
        name: 'Carbon Dioxide',
        earthSimilarity: 0.3,
        description: 'Venus-like atmosphere, dense CO₂',
        color: '#FF9800'
    },
    'hydrogen-helium': {
        name: 'Hydrogen-Helium',
        earthSimilarity: 0.1,
        description: 'Gas giant atmosphere, primarily H₂ and He',
        color: '#2196F3'
    },
    'methane': {
        name: 'Methane',
        earthSimilarity: 0.2,
        description: 'Titan-like atmosphere with methane and nitrogen',
        color: '#9C27B0'
    },
    'thin': {
        name: 'Thin or No Atmosphere',
        earthSimilarity: 0.05,
        description: 'Minimal atmospheric pressure',
        color: '#795548'
    },
    'unknown': {
        name: 'Unknown/Other',
        earthSimilarity: 0.5,
        description: 'Composition not determined',
        color: '#607D8B'
    }
};

// Habitability calculation algorithm
function calculateHabitability(params) {
    // Normalize parameters
    const normalized = normalizeParameters(params);
    
    // Calculate individual scores
    const scores = {
        temperature: calculateTemperatureScore(normalized.planetTemp),
        size: calculateSizeScore(normalized.planetRadius, normalized.planetMass),
        orbit: calculateOrbitScore(normalized.orbitalPeriod, normalized.starTemp),
        stellar: calculateStellarScore(normalized.starTemp, normalized.starRadius),
        atmosphere: calculateAtmosphereScore(normalized.atmosphericComposition),
        water: calculateWaterScore(normalized.waterPresence),
        magnetic: calculateMagneticScore(normalized.magneticField)
    };
    
    // Weighted average
    const weights = {
        temperature: 0.25,
        size: 0.20,
        orbit: 0.15,
        stellar: 0.15,
        atmosphere: 0.10,
        water: 0.10,
        magnetic: 0.05
    };
    
    let totalScore = 0;
    let totalWeight = 0;
    
    for (const [key, score] of Object.entries(scores)) {
        totalScore += score * weights[key];
        totalWeight += weights[key];
    }
    
    const finalScore = Math.round((totalScore / totalWeight) * 100);
    
    // Determine habitability class
    let habitabilityClass = 'Unlikely Habitable';
    for (const [className, info] of Object.entries(HABITABILITY_CLASSES)) {
        if (finalScore >= info.minScore) {
            habitabilityClass = className;
        } else {
            break;
        }
    }
    
    return {
        habitabilityScore: finalScore,
        habitabilityClass: habitabilityClass,
        componentScores: scores,
        normalizedParams: normalized
    };
}

function normalizeParameters(params) {
    return {
        planetRadius: Math.min(Math.max(params.planetRadius || 1, 0.1), 20),
        planetMass: Math.min(Math.max(params.planetMass || 1, 0.1), 10),
        planetTemp: Math.min(Math.max(params.planetTemp || 288, 0), 1000),
        orbitalPeriod: Math.min(Math.max(params.orbitalPeriod || 365, 0.1), 10000),
        starRadius: Math.min(Math.max(params.starRadius || 1, 0.01), 1000),
        starTemp: Math.min(Math.max(params.starTemp || 5778, 1000), 50000),
        stellarLuminosity: params.stellarLuminosity || calculateLuminosity(params.starTemp, params.starRadius),
        atmosphericComposition: params.atmosphericComposition || 'unknown',
        waterPresence: params.waterPresence || 'maybe',
        magneticField: Math.min(Math.max(params.magneticField || 1, 0), 10)
    };
}

function calculateLuminosity(starTemp, starRadius) {
    // L ∝ R² T⁴
    const solarTemp = 5778;
    const solarRadius = 1;
    return Math.pow(starRadius / solarRadius, 2) * Math.pow(starTemp / solarTemp, 4);
}

function calculateTemperatureScore(temp) {
    // Optimal range for liquid water: 273-373 K
    if (temp >= 273 && temp <= 373) {
        const optimal = 293; // 20°C
        const deviation = Math.abs(temp - optimal);
        return Math.max(0, 1 - deviation / 100);
    }
    return Math.max(0, 1 - Math.abs(temp - 293) / 200);
}

function calculateSizeScore(radius, mass) {
    // Earth-like size preference
    const radiusScore = Math.exp(-Math.pow((radius - 1) / 0.5, 2));
    const massScore = Math.exp(-Math.pow((mass - 1) / 2, 2));
    return (radiusScore + massScore) / 2;
}

function calculateOrbitScore(period, starTemp) {
    // Check if in habitable zone (simplified)
    const distance = calculatePlanetDistance(period);
    const habitableZone = calculateHabitableZone(starTemp);
    const zoneStatus = checkHabitableZone(distance, habitableZone);
    
    if (zoneStatus.status === 'in-hz') return 1.0;
    if (zoneStatus.status.includes('optimistic')) return 0.7;
    return 0.3;
}

function calculateStellarScore(starTemp, starRadius) {
    // Prefer G-type stars
    let score = 0.5;
    
    if (starTemp >= 5200 && starTemp <= 6000) {
        // G-type
        score = 1.0;
    } else if (starTemp >= 3700 && starTemp <= 5200) {
        // K-type
        score = 0.8;
    } else if (starTemp >= 6000 && starTemp <= 7500) {
        // F-type
        score = 0.7;
    } else if (starTemp >= 2400 && starTemp <= 3700) {
        // M-type
        score = 0.6;
    }
    
    // Penalize very large stars
    if (starRadius > 10) score *= 0.5;
    
    return score;
}

function calculateAtmosphereScore(composition) {
    const comp = ATMOSPHERIC_COMPOSITIONS[composition] || ATMOSPHERIC_COMPOSITIONS.unknown;
    return comp.earthSimilarity;
}

function calculateWaterScore(presence) {
    switch(presence) {
        case 'yes': return 1.0;
        case 'maybe': return 0.6;
        case 'no': return 0.2;
        default: return 0.5;
    }
}

function calculateMagneticScore(field) {
    // Optimal around Earth's field strength
    return Math.exp(-Math.pow((field - 1) / 2, 2));
}

// Helper function to calculate planet distance from orbital period
function calculatePlanetDistance(orbitalPeriod, starMass = 1) {
    const periodYears = orbitalPeriod / 365.25;
    return Math.cbrt(periodYears * periodYears * starMass);
}

// Helper function from main.js (repeated for convenience)
function calculateHabitableZone(starTemp, starLuminosity = 1) {
    const Ts = starTemp - 5780;
    let innerEdge = Math.sqrt(starLuminosity / 1.1);
    let outerEdge = Math.sqrt(starLuminosity / 0.53);
    innerEdge *= (1 + 0.0001 * Ts + 0.000005 * Ts * Ts);
    outerEdge *= (1 + 0.0001 * Ts + 0.000005 * Ts * Ts);
    return {
        inner: Math.max(0.1, innerEdge),
        outer: outerEdge,
        optimisticInner: Math.sqrt(starLuminosity / 1.8),
        optimisticOuter: Math.sqrt(starLuminosity / 0.28)
    };
}

function checkHabitableZone(planetDistance, habitableZone) {
    if (planetDistance < habitableZone.inner) {
        return { status: 'too-hot', distance: 'inside inner edge' };
    } else if (planetDistance > habitableZone.outer) {
        return { status: 'too-cold', distance: 'outside outer edge' };
    } else if (planetDistance < habitableZone.optimisticInner) {
        return { status: 'optimistic-hot', distance: 'in optimistic HZ (hot)' };
    } else if (planetDistance > habitableZone.optimisticOuter) {
        return { status: 'optimistic-cold', distance: 'in optimistic HZ (cold)' };
    } else {
        return { status: 'in-hz', distance: 'within conservative HZ' };
    }
}

// Get exoplanet by ID
function getExoplanetById(id) {
    return EXOPLANET_DATABASE.find(planet => planet.id === id);
}

// Get exoplanets by filters
function filterExoplanets(filters = {}) {
    return EXOPLANET_DATABASE.filter(planet => {
        for (const [key, value] of Object.entries(filters)) {
            if (value !== undefined && planet[key] !== value) {
                return false;
            }
        }
        return true;
    });
}

// Sort exoplanets
function sortExoplanets(planets, sortBy = 'habitabilityScore', ascending = false) {
    return [...planets].sort((a, b) => {
        let aValue = a[sortBy];
        let bValue = b[sortBy];
        
        if (typeof aValue === 'string') {
            aValue = aValue.toLowerCase();
            bValue = bValue.toLowerCase();
        }
        
        if (aValue < bValue) return ascending ? -1 : 1;
        if (aValue > bValue) return ascending ? 1 : -1;
        return 0;
    });
}

// Generate random exoplanet data for testing
function generateRandomExoplanet() {
    const starTypes = Object.keys(STAR_TYPES);
    const atmospheres = Object.keys(ATMOSPHERIC_COMPOSITIONS);
    const waterOptions = ['yes', 'maybe', 'no'];
    
    const randomStarType = starTypes[Math.floor(Math.random() * starTypes.length)];
    const starTempRange = STAR_TYPES[randomStarType].tempRange;
    const starTemp = starTempRange[0] + Math.random() * (starTempRange[1] - starTempRange[0]);
    
    const params = {
        planetRadius: 0.5 + Math.random() * 3.5,
        planetMass: 0.3 + Math.random() * 9.7,
        planetTemp: 200 + Math.random() * 300,
        orbitalPeriod: 1 + Math.random() * 999,
        starRadius: 0.1 + Math.random() * 9.9,
        starTemp: starTemp,
        stellarLuminosity: 0.001 + Math.random() * 99.999,
        atmosphericComposition: atmospheres[Math.floor(Math.random() * atmospheres.length)],
        waterPresence: waterOptions[Math.floor(Math.random() * waterOptions.length)],
        magneticField: Math.random() * 5
    };
    
    const result = calculateHabitability(params);
    
    return {
        ...params,
        id: 'random-' + Date.now(),
        name: 'Random Exoplanet ' + Math.floor(Math.random() * 1000),
        discoveryYear: 2000 + Math.floor(Math.random() * 24),
        starType: randomStarType + '-type',
        starName: 'Random Star',
        distance: (10 + Math.random() * 990).toFixed(1) + ' ly',
        habitabilityScore: result.habitabilityScore,
        habitabilityClass: result.habitabilityClass,
        tempSuitability: Math.round(result.componentScores.temperature * 100),
        atmosphericLikelihood: Math.round(result.componentScores.atmosphere * 100),
        waterProbability: Math.round(result.componentScores.water * 100),
        radiationTolerance: Math.round(result.componentScores.magnetic * 100),
        description: 'Randomly generated exoplanet for testing purposes.',
        interestingFact: 'This planet was generated by our algorithm for demonstration.',
        imageColor: '#' + Math.floor(Math.random() * 16777215).toString(16)
    };
}

// Export data for external use
window.ExoplanetData = {
    EXOPLANET_DATABASE,
    EXAMPLE_CONFIGURATIONS,
    HABITABILITY_CLASSES,
    STAR_TYPES,
    ATMOSPHERIC_COMPOSITIONS,
    calculateHabitability,
    getExoplanetById,
    filterExoplanets,
    sortExoplanets,
    generateRandomExoplanet
};