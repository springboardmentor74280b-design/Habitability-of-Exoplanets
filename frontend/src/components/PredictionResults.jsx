import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

const CLASS_INFO = {
  0: {
    label: 'Non-Habitable',
    color: '#FF6B6B',
    description: 'This exoplanet shows conditions unsuitable for life as we know it.',
    icon: '❄️',
  },
  1: {
    label: 'Habitable',
    color: '#00F2FF',
    description: 'This planet shows promising conditions for habitability.',
    icon: '🌍',
  },
  2: {
    label: 'Likely Habitable',
    color: '#00FF88',
    description: 'This planet shows high similarity to Earth-like conditions.',
    icon: '✨',
  },
}

export default function PredictionResults({ prediction }) {
  const [gaugeValue, setGaugeValue] = useState(0)
  // Handle both response formats
  const predictionClass = prediction.prediction !== undefined 
    ? prediction.prediction 
    : (prediction.habitability_class === 'Non-Habitable' ? 0 
       : prediction.habitability_class === 'Habitable' ? 1 
       : prediction.habitability_class === 'Likely Habitable' ? 2 : 0)
  
  const classInfo = CLASS_INFO[predictionClass] || CLASS_INFO[0]
  const confidence = prediction.confidence || 0

  useEffect(() => {
    const timer = setTimeout(() => {
      setGaugeValue(confidence * 100)
    }, 100)
    return () => clearTimeout(timer)
  }, [confidence])

  const getGaugeColor = () => {
    if (confidence < 0.5) return '#FF6B6B'
    if (confidence < 0.75) return '#FFD700'
    return '#00FF88'
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="glassmorphic rounded-3xl p-8 lg:p-12"
    >
      <h2 className="text-3xl lg:text-4xl font-orbitron font-bold mb-8 text-center text-neon-cyan text-glow">
        Analysis Results
      </h2>

      {/* Main Result Card */}
      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Habitability Score Gauge */}
        <div className="glassmorphic rounded-2xl p-6 border border-neon-cyan/30">
          <h3 className="text-xl font-orbitron font-semibold mb-4 text-center text-neon-cyan">
            Habitability Score
          </h3>
          
          <div className="relative w-48 h-48 mx-auto">
            {/* Gauge Background */}
            <svg className="transform -rotate-90 w-full h-full">
              <circle
                cx="96"
                cy="96"
                r="80"
                stroke="rgba(0, 242, 255, 0.1)"
                strokeWidth="12"
                fill="none"
              />
              <motion.circle
                cx="96"
                cy="96"
                r="80"
                stroke={getGaugeColor()}
                strokeWidth="12"
                fill="none"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: confidence }}
                transition={{ duration: 1.5, ease: 'easeOut' }}
                strokeDasharray={`${2 * Math.PI * 80}`}
              />
            </svg>
            
            {/* Center Text */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.5, type: 'spring' }}
                className="text-4xl font-orbitron font-bold"
                style={{ color: getGaugeColor() }}
              >
                {Math.round(confidence * 100)}%
              </motion.div>
              <div className="text-sm text-gray-400 font-inter mt-1">
                Confidence
              </div>
            </div>
          </div>
        </div>

        {/* Classification Result */}
        <div className="glassmorphic rounded-2xl p-6 border border-neon-cyan/30">
          <h3 className="text-xl font-orbitron font-semibold mb-4 text-center text-neon-cyan">
            Classification
          </h3>
          
          <div className="flex flex-col items-center justify-center h-full">
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
              className="text-6xl mb-4"
            >
              {classInfo.icon}
            </motion.div>
            
            <motion.h4
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="text-3xl font-orbitron font-bold mb-3 text-center"
              style={{ color: classInfo.color }}
            >
              {prediction.habitability_class || classInfo.label}
            </motion.h4>
            
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
              className="text-gray-300 font-inter text-center"
            >
              {classInfo.description}
            </motion.p>
          </div>
        </div>
      </div>

      {/* Technical Specs Footer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="glassmorphic rounded-xl p-6 border border-stellar-gold/30"
      >
        <h4 className="text-lg font-orbitron font-semibold mb-4 text-center text-stellar-gold">
          Technical Specifications
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-orbitron font-bold text-neon-cyan">0.82</div>
            <div className="text-xs text-gray-400 font-inter mt-1">MCC Score</div>
          </div>
          <div>
            <div className="text-2xl font-orbitron font-bold text-neon-cyan">High</div>
            <div className="text-xs text-gray-400 font-inter mt-1">ROC-AUC</div>
          </div>
          <div>
            <div className="text-2xl font-orbitron font-bold text-neon-cyan">SVM</div>
            <div className="text-xs text-gray-400 font-inter mt-1">Model Type</div>
          </div>
          <div>
            <div className="text-2xl font-orbitron font-bold text-neon-cyan">3-Class</div>
            <div className="text-xs text-gray-400 font-inter mt-1">Classification</div>
          </div>
        </div>
        <p className="text-center text-sm text-gray-400 font-inter mt-4">
          Model: Class-weighted SVM (3-class) | Trained on NASA Exoplanet Archive Data
        </p>
      </motion.div>
    </motion.div>
  )
}
