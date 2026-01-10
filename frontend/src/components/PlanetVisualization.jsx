import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

export default function PlanetVisualization({ radius, density }) {
  const [rotation, setRotation] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation(prev => (prev + 1) % 360)
    }, 50)
    return () => clearInterval(interval)
  }, [])

  // Scale planet size based on radius (normalized to viewport)
  const planetSize = Math.min(Math.max(radius * 40, 80), 300)
  
  // Color based on density (blue for water-like, red for dense)
  const densityColor = density < 3 
    ? '#00F2FF' // Water-like (cyan)
    : density < 6 
    ? '#00FF88' // Earth-like (green-cyan)
    : '#FF6B6B' // Dense (red)

  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {/* Orbital Ring */}
      <motion.div
        className="absolute border border-neon-cyan/20 rounded-full"
        style={{
          width: planetSize * 2.5,
          height: planetSize * 2.5,
        }}
        animate={{ rotate: 360 }}
        transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
      />

      {/* Planet */}
      <motion.div
        className="relative rounded-full glow-effect"
        style={{
          width: planetSize,
          height: planetSize,
          background: `radial-gradient(circle at 30% 30%, 
            ${densityColor}88, 
            ${densityColor}44, 
            ${densityColor}22)`,
          boxShadow: `
            0 0 ${planetSize / 2}px ${densityColor}88,
            inset -${planetSize / 4}px -${planetSize / 4}px ${planetSize / 2}px rgba(0, 0, 0, 0.5)
          `,
        }}
        animate={{
          rotate: rotation,
          scale: [1, 1.05, 1],
        }}
        transition={{
          rotate: { duration: 10, repeat: Infinity, ease: 'linear' },
          scale: { duration: 4, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        {/* Surface Details */}
        <div className="absolute inset-0 rounded-full overflow-hidden">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full opacity-30"
              style={{
                width: `${Math.random() * 40 + 10}%`,
                height: `${Math.random() * 40 + 10}%`,
                left: `${Math.random() * 80}%`,
                top: `${Math.random() * 80}%`,
                background: densityColor,
              }}
              animate={{
                opacity: [0.2, 0.4, 0.2],
              }}
              transition={{
                duration: Math.random() * 3 + 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>

        {/* Atmosphere Glow */}
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{
            background: `radial-gradient(circle, transparent 60%, ${densityColor}22 100%)`,
          }}
          animate={{
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </motion.div>

      {/* Info Text */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute bottom-20 left-1/2 transform -translate-x-1/2 text-center"
      >
        <p className="text-neon-cyan font-orbitron text-sm mb-1">
          Radius: {radius.toFixed(2)} R⊕
        </p>
        <p className="text-stellar-gold font-orbitron text-sm">
          Density: {density.toFixed(2)} g/cm³
        </p>
      </motion.div>

      {/* Stars Background Effect */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-neon-cyan rounded-full"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            opacity: [0.2, 1, 0.2],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: Math.random() * 2 + 1,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  )
}
