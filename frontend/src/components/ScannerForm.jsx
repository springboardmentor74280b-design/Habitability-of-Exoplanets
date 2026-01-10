import { motion } from 'framer-motion'
import { useState } from 'react'

const FEATURE_CONFIG = [
  {
    name: 'planet_radius',
    label: 'Planet Radius',
    unit: 'Earth radii',
    min: 0.1,
    max: 20,
    step: 0.1,
    description: 'Radius of the exoplanet relative to Earth',
  },
  {
    name: 'planet_mass',
    label: 'Planet Mass',
    unit: 'Earth masses',
    min: 0.01,
    max: 50,
    step: 0.1,
    description: 'Mass of the exoplanet relative to Earth',
  },
  {
    name: 'orbital_period',
    label: 'Orbital Period',
    unit: 'days',
    min: 1,
    max: 10000,
    step: 1,
    description: 'Time taken for one complete orbit',
  },
  {
    name: 'stellar_temperature',
    label: 'Stellar Temperature',
    unit: 'Kelvin',
    min: 2000,
    max: 10000,
    step: 10,
    description: 'Effective temperature of the host star',
  },
  {
    name: 'stellar_luminosity',
    label: 'Stellar Luminosity',
    unit: 'Solar luminosities',
    min: 0.001,
    max: 100,
    step: 0.01,
    description: 'Luminosity of the host star relative to the Sun',
  },
  {
    name: 'planet_density',
    label: 'Planet Density',
    unit: 'g/cm³',
    min: 0.5,
    max: 20,
    step: 0.1,
    description: 'Average density of the exoplanet',
  },
  {
    name: 'semi_major_axis',
    label: 'Semi-Major Axis',
    unit: 'AU',
    min: 0.01,
    max: 10,
    step: 0.01,
    description: 'Average distance from the host star',
  },
]

export default function ScannerForm({ formData, onInputChange, onPredict, loading }) {
  const [expandedFeature, setExpandedFeature] = useState(null)

  const getValuePercentage = (name) => {
    const config = FEATURE_CONFIG.find(f => f.name === name)
    if (!config) return 0
    return ((formData[name] - config.min) / (config.max - config.min)) * 100
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="glassmorphic rounded-3xl p-6 lg:p-10"
    >
      <h2 className="text-3xl lg:text-4xl font-orbitron font-bold mb-8 text-center text-neon-cyan text-glow">
        Planetary Scanner
      </h2>

      <div className="space-y-6">
        {FEATURE_CONFIG.map((feature, index) => (
          <motion.div
            key={feature.name}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glassmorphic rounded-xl p-5 border border-neon-cyan/30 hover:border-neon-cyan/60 transition-all"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex-1">
                <label className="text-lg font-orbitron font-semibold text-neon-cyan mb-1 block">
                  {feature.label}
                </label>
                <p className="text-sm text-gray-400 font-inter">{feature.description}</p>
              </div>
              <div className="ml-4 text-right">
                <input
                  type="number"
                  value={formData[feature.name]}
                  onChange={(e) => onInputChange(feature.name, e.target.value)}
                  min={feature.min}
                  max={feature.max}
                  step={feature.step}
                  className="w-32 px-4 py-2 bg-space-blue/50 border border-neon-cyan/50 
                           rounded-lg text-neon-cyan font-orbitron text-center
                           focus:outline-none focus:border-neon-cyan focus:glow-effect
                           transition-all"
                />
                <span className="block text-xs text-gray-400 mt-1 font-inter">
                  {feature.unit}
                </span>
              </div>
            </div>

            {/* Slider */}
            <div className="relative">
              <input
                type="range"
                min={feature.min}
                max={feature.max}
                step={feature.step}
                value={formData[feature.name]}
                onChange={(e) => onInputChange(feature.name, e.target.value)}
                className="w-full h-2 bg-space-blue rounded-lg appearance-none cursor-pointer
                         slider-thumb"
                style={{
                  background: `linear-gradient(to right, 
                    #00F2FF 0%, 
                    #00F2FF ${getValuePercentage(feature.name)}%, 
                    rgba(0, 242, 255, 0.2) ${getValuePercentage(feature.name)}%, 
                    rgba(0, 242, 255, 0.2) 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1 font-inter">
                <span>{feature.min}</span>
                <span>{feature.max}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Predict Button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onPredict}
        disabled={loading}
        className="w-full mt-8 py-4 bg-gradient-to-r from-galactic-purple via-neon-cyan to-galactic-purple
                 rounded-xl font-orbitron font-bold text-space-blue text-xl
                 glow-effect hover:glow-effect transition-all duration-300
                 shadow-lg hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed
                 bg-[length:200%_100%] hover:bg-[position:100%_0]"
      >
        {loading ? (
          <span className="flex items-center justify-center">
            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-space-blue" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Calculating Trajectory...
          </span>
        ) : (
          'Analyze Habitability'
        )}
      </motion.button>
    </motion.div>
  )
}
