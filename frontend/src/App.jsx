import { useState } from 'react'
import { motion } from 'framer-motion'
import Hero from './components/Hero'
import ScannerForm from './components/ScannerForm'
import PlanetVisualization from './components/PlanetVisualization'
import PredictionResults from './components/PredictionResults'
import { predictHabitability } from './api/predict'

function App() {
  const [formData, setFormData] = useState({
    planet_radius: 1.0,
    planet_mass: 1.0,
    orbital_period: 365.0,
    stellar_temperature: 5778.0,
    stellar_luminosity: 1.0,
    planet_density: 5.5,
    semi_major_axis: 1.0,
  })
  
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleInputChange = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }))
    setPrediction(null)
    setError(null)
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setPrediction(null)
    
    try {
      const result = await predictHabitability(formData)
      setPrediction(result)
    } catch (err) {
      setError(err.message || 'Failed to predict habitability')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-space-blue relative overflow-hidden">
      {/* Animated Starfield Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-space-blue via-galactic-purple/20 to-space-blue"></div>
        {[...Array(100)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-neon-cyan rounded-full"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: Math.random(),
            }}
            animate={{
              opacity: [0.3, 1, 0.3],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Split Screen Layout */}
      <div className="relative z-10 flex min-h-screen">
        {/* Left Side - Fixed 3D Animation */}
        <div className="hidden lg:flex lg:w-1/2 xl:w-2/5 fixed left-0 top-0 h-full items-center justify-center">
          <PlanetVisualization 
            radius={formData.planet_radius}
            density={formData.planet_density}
          />
        </div>

        {/* Right Side - Scrollable Content */}
        <div className="w-full lg:w-1/2 xl:w-3/5 lg:ml-auto">
          <div className="container mx-auto px-4 py-8 lg:py-12">
            {/* Hero Section */}
            <Hero onScanClick={() => {
              document.getElementById('scanner').scrollIntoView({ behavior: 'smooth' })
            }} />

            {/* Scanner Form Section */}
            <div id="scanner" className="mt-16">
              <ScannerForm
                formData={formData}
                onInputChange={handleInputChange}
                onPredict={handlePredict}
                loading={loading}
              />
            </div>

            {/* Prediction Results */}
            {prediction && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="mt-12"
              >
                <PredictionResults prediction={prediction} />
              </motion.div>
            )}

            {/* Error Display */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 glassmorphic rounded-2xl p-6 border-red-500/50"
              >
                <p className="text-red-400 text-center font-orbitron">
                  ⚠️ {error}
                </p>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
