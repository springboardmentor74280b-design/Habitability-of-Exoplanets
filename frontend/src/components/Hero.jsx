import { motion } from 'framer-motion'

export default function Hero({ onScanClick }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      className="glassmorphic rounded-3xl p-8 lg:p-12 mb-12"
    >
      <div className="text-center">
        <motion.h1
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="text-5xl lg:text-7xl font-orbitron font-bold mb-4 text-glow"
          style={{ color: '#00F2FF' }}
        >
          EXOSCAN
        </motion.h1>
        
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="text-xl lg:text-2xl font-orbitron mb-2 text-stellar-gold"
        >
          Deep Space Exoplanet Classifier
        </motion.p>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6 }}
          className="glassmorphic rounded-2xl p-6 mt-8 mb-6"
        >
          <h2 className="text-2xl lg:text-3xl font-orbitron font-semibold mb-4 text-neon-cyan">
            Exoplanet Habitability Analyzer
          </h2>
          <p className="text-gray-300 font-inter text-lg mb-6">
            Analyze exoplanetary conditions using advanced machine learning.
            Powered by Class-weighted SVM with 82% MCC accuracy.
          </p>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onScanClick}
            className="px-8 py-4 bg-gradient-to-r from-neon-cyan to-galactic-purple 
                     rounded-xl font-orbitron font-bold text-space-blue text-lg
                     glow-effect hover:glow-effect transition-all duration-300
                     shadow-lg hover:shadow-2xl"
          >
            Scan the Stars →
          </motion.button>
        </motion.div>
      </div>
    </motion.div>
  )
}
