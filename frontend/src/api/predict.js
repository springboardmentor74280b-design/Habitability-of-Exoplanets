import axios from 'axios'

const API_BASE_URL = 'http://127.0.0.1:8000/api'

/**
 * Prepares the request payload with named features.
 * The backend now accepts this format and maps internally.
 */
function preparePayload(features) {
  return {
    features: features
  }
}

export async function predictHabitability(features) {
  try {
    const payload = preparePayload(features)
    
    const response = await axios.post(`${API_BASE_URL}/predict`, payload, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    })
    
    return response.data
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Prediction failed')
    } else if (error.request) {
      throw new Error('Unable to connect to the server. Please ensure the backend is running.')
    } else {
      throw new Error(error.message || 'An unexpected error occurred')
    }
  }
}
