const { useState, useEffect } = React;

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// --- Types & Data ---
const PLANETS = [
    { name: 'Mercury', image: 'https://images.unsplash.com/photo-1614732414444-096e5f1122d5?w=400&h=400&fit=crop', icon: '🌑', color: '#A5A5A5', details: 'The smallest planet and closest to the Sun. It has a cratered surface and almost no atmosphere to retain heat, leading to extreme temperature variations.' },
    { name: 'Venus', image: 'https://images.unsplash.com/photo-1614313913007-2b4ae8ce32d6?w=400&h=400&fit=crop', icon: '🌕', color: '#E3BB76', details: 'Often called Earth\'s twin in size, but has a thick atmosphere that traps heat in a runaway greenhouse effect, making it the hottest planet.' },
    { name: 'Earth', image: 'https://images.unsplash.com/photo-1614730321146-b6fa6a46bac4?w=400&h=400&fit=crop', icon: '🌍', color: '#2271B3', details: 'The only known world with liquid water on its surface and a nitrogen-oxygen atmosphere that supports a vast diversity of life.' },
    { name: 'Mars', image: 'https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=400&h=400&fit=crop', icon: '🔴', color: '#E27B58', details: 'The Red Planet. It features iron-rich dust, polar ice caps, and signs that liquid water once flowed across its now-arid surface.' },
    { name: 'Jupiter', image: 'https://images.unsplash.com/photo-1630839437035-dac17da580d0?w=400&h=400&fit=crop', icon: '🟠', color: '#D39C7E', details: 'The largest planet in our solar system. A gas giant made mostly of hydrogen and helium with a massive storm called the Great Red Spot.' },
    { name: 'Saturn', image: 'https://images.unsplash.com/photo-1614732484003-ef9881555dc3?w=400&h=400&fit=crop', icon: '🪐', color: '#C5AB6E', details: 'Famous for its spectacular ring system composed of ice and rock. Like Jupiter, it is a gas giant with many moons.' },
    { name: 'Uranus', image: 'https://images.unsplash.com/photo-1614730321146-b6fa6a46bac4?w=400&h=400&fit=crop', icon: '🧊', color: '#BBE1E4', details: 'An ice giant that orbits on its side. It has a unique blue-green color due to methane gas in its atmosphere.' },
    { name: 'Neptune', image: 'https://images.unsplash.com/photo-1614732414444-096e5f1122d5?w=400&h=400&fit=crop', icon: '🔵', color: '#6081FF', details: 'The most distant planet. It is an ice giant known for its supersonic winds and deep blue color.' },
];

const HUMAN_SURVIVAL_BASELINE = [
    { label: 'Atmospheric Oxygen (21%)', value: 100 },
    { label: 'Surface Pressure (101.3 kPa)', value: 100 },
    { label: 'Temperature Tolerance (288K)', value: 95 },
    { label: 'Gravitational Baseline (1.0g)', value: 85 },
    { label: 'Magnetospheric Protection', value: 90 },
];

// --- Mock Data Generator (Fallback) ---
const PLANET_DATA = {
    'Mercury': { temp: '440 K', gravity: '0.38 g', atm: 'None (Trace)', dist: '0.39 AU', orbit: '88 days', rad: 'Extreme', score: 5 },
    'Venus': { temp: '737 K', gravity: '0.90 g', atm: 'CO2 (92 bar)', dist: '0.72 AU', orbit: '225 days', rad: 'High', score: 10 },
    'Earth': { temp: '288 K', gravity: '1.00 g', atm: 'N2 (78%) / O2 (21%)', dist: '1.00 AU', orbit: '365 days', rad: 'Low (Safe)', score: 100 },
    'Mars': { temp: '210 K', gravity: '0.38 g', atm: 'CO2 (0.006 bar)', dist: '1.52 AU', orbit: '687 days', rad: 'High', score: 45 },
    'Jupiter': { temp: '165 K', gravity: '2.53 g', atm: 'H2 / He (Gas)', dist: '5.20 AU', orbit: '11.9 years', rad: 'Deadly', score: 2 },
    'Saturn': { temp: '134 K', gravity: '1.06 g', atm: 'H2 / He (Gas)', dist: '9.58 AU', orbit: '29.5 years', rad: 'High', score: 3 },
    'Uranus': { temp: '76 K', gravity: '0.89 g', atm: 'H2 / He / CH4', dist: '19.22 AU', orbit: '84 years', rad: 'Moderate', score: 5 },
    'Neptune': { temp: '72 K', gravity: '1.14 g', atm: 'H2 / He / CH4', dist: '30.05 AU', orbit: '165 years', rad: 'Moderate', score: 4 }
};

function generateMockData(source, target) {
    const sData = PLANET_DATA[source];
    const tData = PLANET_DATA[target];
    
    return {
        metrics: {
            temperature: { source: sData.temp, target: tData.temp, delta: "Diff", score: tData.score },
            gravity: { source: sData.gravity, target: tData.gravity, delta: "Diff", score: tData.score },
            atmosphere: { source: sData.atm, target: tData.atm, delta: "Diff", score: tData.score },
            distance: { source: sData.dist, target: tData.dist, delta: "Diff", score: tData.score },
            orbit: { source: sData.orbit, target: tData.orbit, delta: "Diff", score: tData.score },
            radiation: { source: sData.rad, target: tData.rad, delta: "Diff", score: tData.score },
        },
        sourceHabitabilityScore: sData.score,
        targetHabitabilityScore: tData.score,
        status: tData.score > 75 ? 'Habitable' : (tData.score > 40 ? 'Moderate' : 'Inhospitable'),
        winner: sData.score >= tData.score ? source : target,
        predictionText: `Comparison logic indicates ${sData.score >= tData.score ? source : target} is significantly more suitable for biological life based on known metrics.`,
        detailedAnalysis: `Based on astronomical data: ${target} has a temperature of ${tData.temp} and gravity of ${tData.gravity}, compared to ${source}'s ${sData.temp} / ${sData.gravity}. The atmospheric composition of ${target} (${tData.atm}) presents major challenges compared to ${source}.`,
    };
}

// --- Components ---

const AnalysisViewer = () => {
    const [selected, setSelected] = useState(0);
    
    const items = [
        {
            title: "Habitability Score Distribution",
            description: "Visual representation of how the calculated habitability scores are distributed across the entire Kepler dataset, highlighting the rarity of high-scoring candidates.",
            src: "https://placehold.co/800x500/1e293b/white?text=Habitability+Score",
            fallback: "Habitability+Score"
        },
        {
            title: "Dataset Sample",
            description: "A snapshot of the raw Kepler Threshold Crossing Events (TCE) table, showing the initial features provided by NASA before processing.",
            src: "https://placehold.co/800x500/1e293b/white?text=Dataset+Sample",
            fallback: "Dataset+Sample"
        },
        {
            title: "Training & Testing Split",
            description: "Visual breakdown of the dataset partitioning. We utilized an 80/20 split to ensure the model had sufficient data to learn while reserving a distinct portion for unbiased validation.",
            src: "https://placehold.co/800x500/1e293b/white?text=Train+Test+Split",
            fallback: "Train+Test+Split"
        },
        {
            title: "Data Cleaning & Engineering",
            description: "Flowchart or summary of the preprocessing pipeline: handling missing values, removing noise, normalization, and the creation of the custom 'Habitable_Class' target variable.",
            src: "https://placehold.co/800x500/1e293b/white?text=Data+Cleaning",
            fallback: "Data+Cleaning"
        },
        {
            title: "Correlation Heatmap",
            description: "A heatmap visualizing the linear relationships between planetary features. Strong correlations (red/blue) help identify redundant variables before training the model.",
            src: "https://raw.githubusercontent.com/ExoHabitAI/assets/main/correlation_heatmap_placeholder.png",
            fallback: "Correlation+Heatmap"
        },
        {
            title: "Population Scatter Plot",
            description: "Planetary Radius vs. Period scatter plot. The highlighted cluster represents the 'Goldilocks Zone' candidates—planets with Earth-like size and orbital periods conducive to liquid water.",
            src: "https://raw.githubusercontent.com/ExoHabitAI/assets/main/radius_period_scatter_placeholder.png",
            fallback: "Population+Scatter"
        },
        {
            title: "Random Forest Results",
            description: "Confusion matrix and detailed classification report for the Random Forest model, displaying precision, recall, and f1-score metrics.",
            src: "https://placehold.co/800x500/1e293b/white?text=Random+Forest+Results",
            fallback: "RF+Results"
        },
        {
            title: "XGBoost Accuracy",
            description: "Performance metrics for the Extreme Gradient Boosting (XGBoost) model, comparing its accuracy against the Random Forest baseline.",
            src: "https://placehold.co/800x500/1e293b/white?text=XGBoost+Accuracy",
            fallback: "XGBoost+Accuracy"
        },
        {
            title: "Feature Importance Plot",
            description: "Derived from the Random Forest classifier, this chart ranks which specific planetary attributes (like Insolation Flux or Star Temp) were most critical in determining the habitability classification.",
            src: "https://placehold.co/800x500/1e293b/white?text=Feature+Importance",
            fallback: "Feature+Importance"
        },
        {
            title: "Top 10 Candidates Ranking",
            description: "The final output of our ML pipeline: a ranked list of the specific Kepler Objects of Interest (KOI) with the highest predicted probability of being habitable.",
            src: "https://placehold.co/800x500/1e293b/white?text=Top+10+Ranking",
            fallback: "Top+10+Ranking"
        }
    ];

    return (
        <div className="bg-slate-900/60 p-8 md:p-10 rounded-[3rem] border border-white/10 shadow-2xl backdrop-blur-md">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
                <div>
                    <h3 className="text-3xl font-bold text-blue-400">Analysis Visualizations</h3>
                    <p className="text-slate-400 text-sm mt-2">Explore the comprehensive data science pipeline.</p>
                </div>
            </div>
            
            <div className="flex flex-col lg:flex-row gap-8">
                <div className="lg:w-1/3 flex flex-col gap-3 max-h-[600px] overflow-y-auto pr-2">
                    {items.map((item, idx) => (
                        <button
                            key={idx}
                            onClick={() => setSelected(idx)}
                            className={`text-left p-5 rounded-2xl border transition-all duration-300 group relative overflow-hidden flex-shrink-0 ${
                                selected === idx 
                                ? 'bg-blue-600 border-blue-500 shadow-lg shadow-blue-900/20' 
                                : 'bg-black/20 border-white/5 hover:bg-white/5 hover:border-white/10'
                            }`}
                        >
                            <div className="relative z-10">
                                <h4 className={`font-bold text-lg mb-1 ${selected === idx ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                                    {item.title}
                                </h4>
                                <div className={`text-xs font-bold uppercase tracking-widest ${selected === idx ? 'text-blue-200' : 'text-slate-500'}`}>
                                    Figure {idx + 1}
                                </div>
                            </div>
                        </button>
                    ))}
                </div>

                <div className="lg:w-2/3 h-[600px]">
                    <div className="bg-black/40 rounded-3xl border border-white/10 p-2 h-full flex flex-col">
                        <div className="relative aspect-video rounded-2xl overflow-hidden bg-[#020408] border border-white/5 flex-1 group">
                            <img 
                                src={items[selected].src} 
                                onError={(e) => e.currentTarget.src=`https://placehold.co/800x500/1e293b/white?text=${items[selected].fallback}`}
                                alt={items[selected].title} 
                                className="w-full h-full object-contain animate-in fade-in duration-500 group-hover:scale-105 transition-transform duration-700"
                                key={selected}
                            />
                        </div>
                        <div className="p-6 bg-slate-900/50 rounded-b-2xl">
                            <h4 className="text-xl font-bold text-white mb-2">{items[selected].title}</h4>
                            <p className="text-slate-400 leading-relaxed text-sm">
                                {items[selected].description}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const CreatePlanetView = () => {
    const [params, setParams] = useState({
        radius: 1.0,
        temp: 288,
        insol: 1.0,
        period: 365,
        steff: 5778,
        sradius: 1.0
    });
    const [result, setResult] = useState(null);
    const [isCalculating, setIsCalculating] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setParams(prev => ({
            ...prev,
            [name]: parseFloat(value) || 0
        }));
    };

    const calculateScore = async () => {
        setIsCalculating(true);
        setResult(null);
        
        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const data = await response.json();
            
            let status, description, colorClass;
            const score = parseFloat(data.score);
            
            if (score >= 70) {
                status = "HIGHLY HABITABLE";
                description = "Excellent conditions for life as we know it! The combination of temperature and planetary size suggests a high likelihood of liquid water.";
                colorClass = "text-emerald-400";
            } else if (score >= 50) {
                status = "MODERATELY HABITABLE";
                description = "Potentially habitable with some challenges. Conditions allow for existence, though extreme weather or atmospheric variations might be present.";
                colorClass = "text-yellow-400";
            } else if (score >= 30) {
                status = "MARGINALLY HABITABLE";
                description = "Limited habitability with significant challenges. Life would likely be microbial or require shielded subterranean environments.";
                colorClass = "text-orange-400";
            } else {
                status = "NOT HABITABLE";
                description = "Conditions are not suitable for life. Extreme temperatures, crushing gravity, or lack of energy sources make this world hostile.";
                colorClass = "text-red-500";
            }

            setResult({ score: score.toFixed(2), status, description, colorClass });
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to get prediction. Make sure the backend server is running.');
        } finally {
            setIsCalculating(false);
        }
    };

    return (
        <div className="max-w-6xl mx-auto py-12 animate-in fade-in">
            <header className="text-center mb-16">
                <h1 className="text-8xl font-black mb-6 tracking-tighter drop-shadow-2xl">ExoHabitAI</h1>
                <p className="text-slate-400 text-xl font-medium mb-8">AI-Powered Planetary Habitability Prediction</p>
                <div className="inline-block border-b-2 border-blue-500 pb-2">
                    <h2 className="text-2xl font-bold text-blue-400 uppercase tracking-widest">Create Your Planet</h2>
                </div>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                <div className="bg-slate-900/60 p-8 rounded-[2.5rem] border border-white/10 shadow-2xl backdrop-blur-md">
                    <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                        <i className="fa-solid fa-sliders text-blue-500"></i> Planetary Parameters
                    </h3>
                    
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-xs font-black uppercase tracking-widest text-slate-500">Planetary Radius (Earth = 1.0)</label>
                                <input type="number" step="0.1" name="radius" value={params.radius} onChange={handleChange} className="w-full bg-black/40 border border-white/10 p-4 rounded-xl font-mono text-white focus:border-blue-500 outline-none transition-colors" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-black uppercase tracking-widest text-slate-500">Equilibrium Temp (Kelvin)</label>
                                <input type="number" step="1" name="temp" value={params.temp} onChange={handleChange} className="w-full bg-black/40 border border-white/10 p-4 rounded-xl font-mono text-white focus:border-blue-500 outline-none transition-colors" />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-xs font-black uppercase tracking-widest text-slate-500">Insolation Flux (Earth = 1.0)</label>
                                <input type="number" step="0.1" name="insol" value={params.insol} onChange={handleChange} className="w-full bg-black/40 border border-white/10 p-4 rounded-xl font-mono text-white focus:border-blue-500 outline-none transition-colors" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-black uppercase tracking-widest text-slate-500">Orbital Period (Days)</label>
                                <input type="number" step="1" name="period" value={params.period} onChange={handleChange} className="w-full bg-black/40 border border-white/10 p-4 rounded-xl font-mono text-white focus:border-blue-500 outline-none transition-colors" />
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-xs font-black uppercase tracking-widest text-slate-500">Star Temp (Kelvin)</label>
                                <input type="number" step="10" name="steff" value={params.steff} onChange={handleChange} className="w-full bg-black/40 border border-white/10 p-4 rounded-xl font-mono text-white focus:border-blue-500 outline-none transition-colors" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-black uppercase tracking-widest text-slate-500">Star Radius (Sun = 1.0)</label>
                                <input type="number" step="0.1" name="sradius" value={params.sradius} onChange={handleChange} className="w-full bg-black/40 border border-white/10 p-4 rounded-xl font-mono text-white focus:border-blue-500 outline-none transition-colors" />
                            </div>
                        </div>

                        <button 
                            onClick={calculateScore} 
                            disabled={isCalculating}
                            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-black py-5 rounded-xl uppercase tracking-widest shadow-lg shadow-blue-900/20 transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed mt-4"
                        >
                            {isCalculating ? <i className="fa-solid fa-circle-notch fa-spin"></i> : "Run Prediction Model"}
                        </button>
                    </div>
                </div>

                <div className="bg-slate-900/60 p-8 rounded-[2.5rem] border border-white/10 shadow-2xl backdrop-blur-md flex flex-col items-center justify-center text-center relative overflow-hidden">
                    {!result ? (
                        <div className="text-slate-500 flex flex-col items-center">
                            <i className="fa-solid fa-planet-ringed text-6xl mb-4 opacity-20"></i>
                            <p className="uppercase tracking-widest text-xs font-bold">Awaiting Input Data</p>
                        </div>
                    ) : (
                        <div className="animate-in fade-in slide-in-from-bottom-10 w-full relative z-10">
                            <div className="absolute top-0 right-0 p-4 opacity-20">
                                <i className="fa-solid fa-chart-network text-9xl"></i>
                            </div>
                            
                            <h3 className="text-sm font-black uppercase tracking-[0.3em] text-slate-400 mb-8">Prediction Result</h3>
                            
                            <div className="mb-8 transform hover:scale-105 transition-transform duration-500">
                                <CircularHabitabilityGauge score={parseFloat(result.score)} label="Habitability Index" />
                            </div>

                            <div className="border-t border-white/10 pt-8 mt-4">
                                <h2 className={`text-4xl font-black mb-4 ${result.colorClass}`}>{result.status}</h2>
                                <p className="text-slate-300 leading-relaxed max-w-md mx-auto">{result.description}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

const CelestialLoader = ({ progress }) => {
    const [loadingText, setLoadingText] = useState("Calibrating Sensors...");
    const tips = [
        "Analyzing Atmospheric Density...",
        "Calculating Gravitational Fluctuations...",
        "Simulating Surface Radiation Levels...",
        "Scanning for Liquid Water Signatures...",
        "Verifying Magnetospheric Shielding...",
        "Cross-referencing Bio-habitability Markers..."
    ];

    useEffect(() => {
        const interval = setInterval(() => {
            setLoadingText(tips[Math.floor(Math.random() * tips.length)]);
        }, 1500);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="fixed inset-0 z-[100] bg-[#020408]/90 backdrop-blur-xl flex flex-col items-center justify-center p-8 animate-in fade-in">
            <div className="relative w-64 h-64 mb-12">
                <div className="absolute inset-0 border border-white/5 rounded-full animate-[spin_8s_linear_infinite]"></div>
                <div className="absolute inset-4 border border-blue-500/20 rounded-full animate-[spin_12s_linear_infinite_reverse]"></div>
                <div className="absolute inset-8 border border-orange-500/10 rounded-full animate-[spin_15s_linear_infinite]"></div>
                
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-20 h-20 bg-gradient-to-tr from-blue-600 to-orange-500 rounded-full shadow-[0_0_60px_rgba(59,130,246,0.6)] animate-pulse"></div>
                </div>

                <div className="absolute inset-0 flex items-center justify-center overflow-hidden rounded-full">
                   <div className="w-full h-[2px] bg-blue-400/50 shadow-[0_0_15px_blue] absolute top-0 animate-[bounce_3s_infinite]"></div>
                </div>
            </div>

            <div className="max-w-md w-full text-center">
                <h2 className="text-3xl font-black mb-2 tracking-tighter text-white uppercase italic">Analyzing Cosmos</h2>
                <p className="text-orange-500 font-black text-[10px] uppercase tracking-[0.4em] mb-10 h-4">{loadingText}</p>
                
                <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden border border-white/10 shadow-inner">
                   <div className="h-full bg-gradient-to-r from-blue-600 via-purple-500 to-orange-500 transition-all duration-300 ease-out" style={{ width: `${progress}%` }}></div>
                </div>
                <div className="flex justify-between mt-4 text-[9px] font-black text-slate-500 uppercase tracking-widest">
                   <span>DATA ACQUISITION</span>
                   <span className="text-white">{progress}% COMPLETE</span>
                </div>
            </div>
        </div>
    );
};

const CircularHabitabilityGauge = ({ score, label }) => {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (score / 100) * circumference;
    
    return (
        <div className="flex flex-col items-center">
            <div className="relative w-40 h-40">
                <svg className="w-full h-full transform -rotate-90">
                    <circle cx="80" cy="80" r={radius} stroke="currentColor" strokeWidth="8" fill="transparent" className="text-slate-800" />
                    <circle 
                        cx="80" cy="80" r={radius} 
                        stroke={score > 80 ? '#10b981' : score > 40 ? '#f59e0b' : '#ef4444'} 
                        strokeWidth="8" 
                        fill="transparent" 
                        strokeDasharray={circumference}
                        strokeDashoffset={offset}
                        strokeLinecap="round"
                        className="transition-all duration-1000 ease-out shadow-lg"
                    />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-4xl font-black text-white">{score}%</span>
                    <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Score</span>
                </div>
            </div>
            <span className="mt-4 text-[10px] font-black uppercase tracking-widest text-slate-400">{label}</span>
        </div>
    );
};

const FeatureBar = ({ label, value }) => (
    <div className="mb-4">
        <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1.5">
            <span>{label}</span>
            <span>{value}%</span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden border border-white/5">
            <div className="h-full bg-blue-500 transition-all duration-1000" style={{ width: `${value}%` }}></div>
        </div>
    </div>
);

const ExploreSpaceView = () => {
    const sections = [
        {
            title: 'Galaxies',
            icon: '🌌',
            items: [
                { name: 'Milky Way', desc: 'Our home spiral galaxy, spanning 100,000 light-years and housing billions of planets.' },
                { name: 'Andromeda', desc: 'The closest large galaxy to ours, expected to collide with the Milky Way in 4 billion years.' },
                { name: 'Black Eye Galaxy', desc: 'Famous for its dark band of absorbing dust in front of its bright nucleus.' }
            ]
        },
        {
            title: 'Planetary Systems',
            icon: '🪐',
            items: [
                { name: 'TRAPPIST-1', desc: 'A star system hosting seven Earth-sized rocky planets, some in the habitable zone.' },
                { name: 'Alpha Centauri', desc: 'The nearest stellar system to the Sun, containing three stars and several rocky worlds.' },
                { name: 'Kepler-186', desc: 'Home to Kepler-186f, the first Earth-size planet found in the habitable zone of another star.' }
            ]
        },
        {
            title: 'Stellar Phenomena',
            icon: '🌟',
            items: [
                { name: 'Supernovae', desc: 'The explosive death of a massive star, releasing immense energy and creating heavy elements.' },
                { name: 'Pulsars', desc: 'Highly magnetized rotating neutron stars that emit beams of electromagnetic radiation.' },
                { name: 'Nebulae', desc: 'Giant clouds of dust and gas where new stars are born from gravity-induced collapse.' }
            ]
        }
    ];

    return (
        <div className="animate-in fade-in duration-1000">
            <header className="mb-20">
                <h1 className="text-8xl font-black mb-6 tracking-tighter drop-shadow-2xl">Deep Space Atlas</h1>
                <p className="text-2xl text-slate-400 max-w-2xl font-medium tracking-tight">An analytical catalog of celestial bodies and cosmic phenomena across the neighborhood.</p>
            </header>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-10 mb-24">
                {sections.map((sec, i) => (
                    <div key={i} className="bg-[#0f172a]/60 backdrop-blur-2xl p-10 rounded-[3rem] border border-white/10 flex flex-col h-full">
                        <div className="text-5xl mb-6">{sec.icon}</div>
                        <h2 className="text-3xl font-black uppercase tracking-tight mb-10 text-white border-b border-white/5 pb-4">{sec.title}</h2>
                        <div className="space-y-10 flex-1">
                            {sec.items.map((item, j) => (
                                <div key={j} className="group border-l-4 border-blue-600/30 pl-6 hover:border-blue-500 transition-all">
                                    <h3 className="font-bold text-blue-400 mb-2 text-xl">{item.name}</h3>
                                    <p className="text-slate-400 leading-relaxed font-medium">{item.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <div className="pt-20 border-t border-white/5">
                <h2 className="text-xs font-black uppercase tracking-[0.5em] text-slate-600 mb-16 text-center">Local System Reconnaissance</h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-10">
                    {PLANETS.map((p, i) => (
                        <div key={i} className="bg-[#1e293b]/30 backdrop-blur-md rounded-[3rem] border border-white/5 p-8 flex flex-col items-center text-center group hover:bg-[#2b5ba3]/10 transition-all shadow-xl">
                            <div className="w-full aspect-square rounded-[2rem] overflow-hidden mb-8 shadow-2xl border-4 border-white/5 group-hover:border-blue-500 transition-all duration-700">
                                <img src={p.image} className="w-full h-full object-cover group-hover:scale-125 transition-transform duration-[3000ms]" />
                            </div>
                            <div className="flex items-center gap-3 mb-4">
                                <span className="text-3xl">{p.icon}</span>
                                <h3 className="text-3xl font-black tracking-tighter text-white">{p.name}</h3>
                            </div>
                            <p className="text-slate-400 text-sm leading-relaxed font-medium mb-8 flex-1">
                                {p.details}
                            </p>
                            <div className="w-full pt-6 border-t border-white/5 flex justify-between items-center text-[9px] font-black uppercase tracking-widest text-slate-500">
                                <span>Solar System</span>
                                <span className="text-blue-500 font-black">{p.name === 'Earth' ? 'Optimal' : 'Experimental'}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

const App = () => {
    const [view, setView] = useState('create');
    const [sourceName, setSourceName] = useState('Earth');
    const [targetName, setTargetName] = useState('Mars');
    const [loading, setLoading] = useState(false);
    const [loadingProgress, setLoadingProgress] = useState(0);
    const [comparison, setComparison] = useState(null);

    const handleCompare = async () => {
        setLoading(true);
        setLoadingProgress(5);
        setComparison(null);
        
        const interval = setInterval(() => {
            setLoadingProgress(prev => {
                if (prev >= 90) return prev;
                return prev + Math.floor(Math.random() * 8) + 2;
            });
        }, 400);

        try {
            const response = await fetch(`${API_BASE_URL}/compare`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source: sourceName,
                    target: targetName
                })
            });

            if (!response.ok) {
                throw new Error('Comparison failed');
            }

            const data = await response.json();
            
            setLoadingProgress(100);
            
            setTimeout(() => {
                setComparison(data);
                setLoading(false);
                clearInterval(interval);
            }, 500);
            
        } catch (err) {
            console.warn("API Error, switching to mock data.", err);
            
            const mockData = generateMockData(sourceName, targetName);
            
            setLoadingProgress(100);
            setTimeout(() => {
                setComparison(mockData);
                setLoading(false);
                clearInterval(interval);
            }, 500);
        }
    };

    return (
        <div className="min-h-screen bg-[#020408] text-white selection:bg-blue-500/30 font-sans">
            
            {loading && <CelestialLoader progress={loadingProgress} />}

            <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=2400')] opacity-60 bg-cover bg-center transition-opacity duration-1000"></div>
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#020408]/70 to-[#020408]"></div>
                <div className="absolute inset-0 opacity-20 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] animate-[pulse_10s_infinite]"></div>
            </div>

            <nav className="fixed top-0 w-full z-50 bg-[#0f172a]/90 backdrop-blur-xl border-b border-white/5 px-10 py-5 flex items-center justify-between">
                <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView('create')}>
                    <div className="w-10 h-10 bg-orange-600 rounded-xl flex items-center justify-center shadow-lg shadow-orange-900/40">
                        <span className="text-xl">🚀</span>
                    </div>
                    <span className="text-2xl font-black italic tracking-tighter">ExoHabitAI</span>
                </div>
                <div className="flex gap-12 text-[11px] font-black uppercase tracking-[0.2em] items-center">
                    <button onClick={() => setView('create')} className={`hover:text-orange-400 transition-colors ${view === 'create' ? 'text-orange-400' : ''}`}>Home</button>
                    <button onClick={() => setView('home')} className={`hover:text-orange-400 transition-colors ${view === 'home' ? 'text-orange-400' : ''}`}>Compare</button>
                    <button onClick={() => setView('about')} className={`hover:text-orange-400 transition-colors ${view === 'about' ? 'text-orange-400' : ''}`}>About</button>
                    <button onClick={() => setView('explore')} className={`hover:text-orange-400 transition-colors ${view === 'explore' ? 'text-orange-400' : ''}`}>Explore</button>
                    <button onClick={() => setView('contact')} className={`hover:text-orange-400 transition-colors ${view === 'contact' ? 'text-orange-400' : ''}`}>Contact</button>
                </div>
            </nav>

            <main className="relative z-10 max-w-7xl mx-auto pt-36 pb-24 px-6">
                {view === 'create' && <CreatePlanetView />}

                {view === 'home' && (
                    <div className="animate-in fade-in duration-1000">
                        <header className="text-center mb-16">
                            <h1 className="text-6xl font-black mb-6 tracking-tighter drop-shadow-2xl">Stellar Metrics</h1>
                            <p className="text-slate-400 text-xl font-medium">Comparative Analysis of Cosmic Living Environments</p>
                        </header>

                        <div className="bg-white/10 backdrop-blur-3xl p-12 rounded-[3rem] border border-white/10 shadow-2xl mb-16 max-w-4xl mx-auto">
                            <div className="grid grid-cols-1 md:grid-cols-11 items-center gap-10 mb-10">
                                <div className="md:col-span-5 space-y-3">
                                    <label className="text-[11px] font-black uppercase tracking-widest text-slate-500">Origin World</label>
                                    <select value={sourceName} onChange={(e) => setSourceName(e.target.value)} className="w-full bg-[#0f172a] border-2 border-white/5 p-5 rounded-2xl font-bold text-xl focus:border-orange-500/50 transition-all outline-none appearance-none cursor-pointer text-white">
                                        {PLANETS.map(p => <option key={p.name} value={p.name}>{p.icon} {p.name}</option>)}
                                    </select>
                                </div>
                                <div className="md:col-span-1 flex justify-center text-orange-500 text-4xl pt-8">⇄</div>
                                <div className="md:col-span-5 space-y-3">
                                    <label className="text-[11px] font-black uppercase tracking-widest text-slate-500">Destination World</label>
                                    <select value={targetName} onChange={(e) => setTargetName(e.target.value)} className="w-full bg-[#0f172a] border-2 border-white/5 p-5 rounded-2xl font-bold text-xl focus:border-orange-500/50 transition-all outline-none appearance-none cursor-pointer text-white">
                                        {PLANETS.map(p => <option key={p.name} value={p.name}>{p.icon} {p.name}</option>)}
                                    </select>
                                </div>
                            </div>
                            <button onClick={handleCompare} disabled={loading} className="w-full bg-orange-600 hover:bg-orange-500 text-white font-black py-6 rounded-2xl shadow-2xl transition-all active:scale-[0.98] flex items-center justify-center gap-4 text-2xl disabled:opacity-50">
                                {loading ? "Initializing Scan..." : "Analyze Habitability"}
                            </button>
                        </div>

                        {comparison && !loading && (
                            <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                                
                                <div className="lg:col-span-4 space-y-10">
                                    <div className="bg-white/5 backdrop-blur-md p-10 rounded-[2.5rem] border border-white/10 flex flex-col items-center gap-12">
                                        <CircularHabitabilityGauge score={comparison.sourceHabitabilityScore} label={`${sourceName} Status`} />
                                        <CircularHabitabilityGauge score={comparison.targetHabitabilityScore} label={`${targetName} Status`} />
                                    </div>
                                    
                                    <div className="bg-white/5 backdrop-blur-md p-10 rounded-[2.5rem] border border-white/10">
                                        <h3 className="text-sm font-black uppercase tracking-widest text-slate-500 mb-8 text-center">Human Survival Baseline</h3>
                                        {HUMAN_SURVIVAL_BASELINE.map((b, i) => (
                                            <FeatureBar key={i} label={b.label} value={b.value} />
                                        ))}
                                        <p className="text-[9px] text-slate-500 mt-6 leading-relaxed italic text-center uppercase tracking-widest font-black">Required thresholds for non-assisted human life sustainment.</p>
                                    </div>
                                </div>

                                <div className="lg:col-span-8 space-y-10">
                                    <div className="bg-[#1e293b]/60 backdrop-blur-3xl p-12 rounded-[3rem] border border-white/10 shadow-2xl relative overflow-hidden group">
                                        <div className="flex items-start justify-between mb-10">
                                            <div className="flex items-start gap-4">
                                                <span className="text-5xl">{PLANETS.find(p => p.name === targetName)?.icon}</span>
                                                <div>
                                                    <h2 className="text-4xl font-black tracking-tight">{targetName} Habitability</h2>
                                                    <span className={`text-xl font-bold ${comparison.targetHabitabilityScore > 70 ? 'text-green-400' : comparison.targetHabitabilityScore > 30 ? 'text-orange-400' : 'text-red-400'}`}>{comparison.status}</span>
                                                </div>
                                            </div>
                                            <div className="bg-orange-500 text-white px-6 py-2 rounded-full text-[10px] font-black uppercase tracking-widest shadow-xl">Analysis Complete</div>
                                        </div>

                                        <div className="flex justify-center mb-12">
                                            <CircularHabitabilityGauge score={comparison.targetHabitabilityScore} label="Calculated Score" />
                                        </div>

                                        <div className="w-full h-4 bg-slate-800 rounded-full overflow-hidden mb-8 border border-white/5">
                                            <div 
                                                className={`h-full transition-all duration-1000 ${comparison.targetHabitabilityScore > 70 ? 'bg-green-500' : comparison.targetHabitabilityScore > 30 ? 'bg-orange-500' : 'bg-red-500'}`} 
                                                style={{ width: `${comparison.targetHabitabilityScore}%` }}
                                            ></div>
                                        </div>

                                        <p className="text-2xl text-slate-200 leading-relaxed font-semibold italic mb-6 border-l-4 border-orange-500 pl-6">
                                            "{comparison.predictionText}"
                                        </p>
                                        <p className="text-slate-400 text-lg leading-relaxed font-medium">
                                            {comparison.detailedAnalysis}
                                        </p>
                                    </div>

                                    <div className="bg-white/5 backdrop-blur-md p-10 rounded-[2.5rem] border border-white/10">
                                        <h3 className="text-sm font-black uppercase tracking-widest text-slate-500 mb-8">Environmental Telemetry Comparison</h3>
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-left">
                                                <thead className="text-[10px] font-black uppercase tracking-widest text-slate-500 border-b border-white/5">
                                                    <tr>
                                                        <th className="pb-6">Requirement</th>
                                                        <th className="pb-6">{sourceName}</th>
                                                        <th className="pb-6 text-center">Result</th>
                                                        <th className="pb-6">{targetName}</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-white/5 text-slate-200">
                                                    {[
                                                        ['Temperature', comparison.metrics.temperature],
                                                        ['Gravity', comparison.metrics.gravity],
                                                        ['Atmosphere', comparison.metrics.atmosphere],
                                                        ['Solar Distance', comparison.metrics.distance],
                                                        ['Orbit Period', comparison.metrics.orbit],
                                                        ['Radiation hazard', comparison.metrics.radiation],
                                                    ].map(([label, m], i) => (
                                                        <tr key={i} className="hover:bg-white/5 transition-colors">
                                                            <td className="py-6 font-bold text-slate-400">{label}</td>
                                                            <td className="py-6 font-mono">{m.source}</td>
                                                            <td className="py-6 text-center font-black text-orange-500 text-xs">{m.delta}</td>
                                                            <td className="py-6 font-mono font-bold text-white">{m.target}</td>
                                                        </tr>
                                                    ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>

                            </div>
                        )}
                    </div>
                )}

                {view === 'explore' && <ExploreSpaceView />}
                
                {view === 'about' && (
                    <div className="max-w-6xl mx-auto py-16 animate-in fade-in px-6">
                        <header className="text-center mb-16">
                            <h2 className="text-6xl font-black mb-6 tracking-tighter drop-shadow-2xl">About ExoHabitAI</h2>
                            <p className="text-slate-300 text-lg leading-relaxed max-w-3xl mx-auto">
                                ExoHabitAI is an interactive cosmic platform that blends AI and astronomy to explore the possibility of life beyond Earth. It allows users to create worlds, compare planets, and journey through space using intelligent models and scientific insights. Designed to educate, inspire, and visualize the universe, ExoHabitAI turns complex space science into an engaging digital experience.
                            </p>
                        </header>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-20">
                            <div className="bg-slate-900/60 p-8 rounded-[2rem] border border-white/10 shadow-xl backdrop-blur-md">
                                <div className="text-4xl mb-4">🌍</div>
                                <h3 className="text-xl font-bold text-white mb-3">Create World</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    The Create World page allows users to design a planet and evaluate its habitability. It uses an AI-based habitability prediction model inspired by Random Forest machine learning, where planetary features like temperature, size, orbit, and star properties are analyzed using scientifically weighted factors to generate a habitability score.
                                </p>
                            </div>

                            <div className="bg-slate-900/60 p-8 rounded-[2rem] border border-white/10 shadow-xl backdrop-blur-md">
                                <div className="text-4xl mb-4">⚖️</div>
                                <h3 className="text-xl font-bold text-white mb-3">Compare Planets</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    The Compare Planets page uses an AI language model to compare two planets based on key habitability parameters such as temperature, gravity, atmosphere, and radiation. It provides clear scores and insights to identify which planet is more suitable for life.
                                </p>
                            </div>

                            <div className="bg-slate-900/60 p-8 rounded-[2rem] border border-white/10 shadow-xl backdrop-blur-md">
                                <div className="text-4xl mb-4">🌌</div>
                                <h3 className="text-xl font-bold text-white mb-3">Explore Space</h3>
                                <p className="text-slate-400 text-sm leading-relaxed">
                                    The Explore Space page offers an educational overview of galaxies, planetary systems, and space phenomena. It presents curated scientific information through interactive visuals, helping users understand the universe in a simple and engaging way.
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {view === 'contact' && (
                    <div className="max-w-xl mx-auto py-20 animate-in fade-in">
                        <div className="bg-[#0f172a]/80 backdrop-blur-2xl p-16 rounded-[3rem] border border-white/5 shadow-2xl text-center">
                            <h2 className="text-4xl font-black mb-6">Uplink Hub</h2>
                            <p className="text-slate-500 mb-10">Connect with the Mission Control Research Outpost.</p>
                            
                            <div className="flex flex-wrap justify-center gap-8">
                                <a href="mailto:poojakoppula4@gmail.com" className="flex flex-col items-center group">
                                    <div className="w-16 h-16 bg-white/5 rounded-2xl flex items-center justify-center text-2xl text-slate-400 group-hover:bg-red-500 group-hover:text-white transition-all shadow-lg border border-white/5 group-hover:scale-110 duration-300">
                                        <i className="fa-solid fa-envelope"></i>
                                    </div>
                                    <span className="mt-4 text-[10px] font-black uppercase tracking-widest text-slate-500 group-hover:text-red-400 transition-colors">Email</span>
                                </a>

                                <a href="https://www.linkedin.com/in/k-pooja-reddy-28p09" target="_blank" rel="noopener noreferrer" className="flex flex-col items-center group">
                                    <div className="w-16 h-16 bg-white/5 rounded-2xl flex items-center justify-center text-2xl text-slate-400 group-hover:bg-[#0077b5] group-hover:text-white transition-all shadow-lg border border-white/5 group-hover:scale-110 duration-300">
                                        <i className="fa-brands fa-linkedin-in"></i>
                                    </div>
                                    <span className="mt-4 text-[10px] font-black uppercase tracking-widest text-slate-500 group-hover:text-[#0077b5] transition-colors">LinkedIn</span>
                                </a>

                                <a href="https://github.com/springboardmentor74280b-design/Habitability-of-Exoplanets/tree/K.POOJA-REDDY" target="_blank" rel="noopener noreferrer" className="flex flex-col items-center group">
                                    <div className="w-16 h-16 bg-white/5 rounded-2xl flex items-center justify-center text-2xl text-slate-400 group-hover:bg-[#333] group-hover:text-white transition-all shadow-lg border border-white/5 group-hover:scale-110 duration-300">
                                        <i className="fa-brands fa-github"></i>
                                    </div>
                                    <span className="mt-4 text-[10px] font-black uppercase tracking-widest text-slate-500 group-hover:text-slate-300 transition-colors">GitHub</span>
                                </a>
                            </div>
                        </div>
                    </div>
                )}
            </main>

            <footer className="relative z-10 py-12 border-t border-white/5 text-center">
                <p className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-700">© 2026 ExoHabitAI • Exploring the Cosmos for Humanity</p>
            </footer>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
