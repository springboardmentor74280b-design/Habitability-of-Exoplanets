import { useState } from "react";


export default function App() {

    const [score, setScore] = useState(null);

    const predict = () => {
        setScore(87);
    };

    return (<div className="container">

        ```
        <h1>🌍 Exoplanet Habitability Prediction</h1>

        <div className="card">

            <input placeholder="Planet Radius" />
            <input placeholder="Planet Mass" />
            <input placeholder="Orbital Period" />
            <input placeholder="Star Temperature" />
            <input placeholder="Star Luminosity" />

            <button onClick={predict}>
                Predict Habitability
            </button>

            {score && (
                <div className="result">
                    <h2>Habitability Score: {score}%</h2>
                    <p>✅ Potentially Habitable Planet</p>
                </div>
            )}

        </div>

    </div>)`` `

);
}
`;
}
