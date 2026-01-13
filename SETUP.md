# Quick Setup Guide

## Step 1: Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows (CMD):
     ```cmd
     venv\Scripts\activate.bat
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Gemini API Key:**
   - Windows (PowerShell):
     ```powershell
     $env:GEMINI_API_KEY="your_api_key_here"
     ```
   - Linux/Mac:
     ```bash
     export GEMINI_API_KEY="your_api_key_here"
     ```

6. **Train the model:**
   ```bash
   python train_model.py
   ```
   
   This will create the model files in `backend/models/` directory.

7. **Start the Flask server:**
   ```bash
   python app.py
   ```
   
   Server will run on `http://localhost:5000`

## Step 2: Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Start a local server:**
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Or using Node.js (if installed)
   npx http-server -p 8000
   ```

3. **Open in browser:**
   ```
   http://localhost:8000
   ```

## Troubleshooting

### Model Training Issues
- Ensure `Kepler_Threshold_Crossing_Events_Table.csv` is in the parent directory or backend directory
- Check that all required Python packages are installed
- Verify you have sufficient memory for training

### API Connection Issues
- Ensure backend server is running on port 5000
- Check CORS settings if accessing from different origin
- Verify API_BASE_URL in `frontend/app.js` matches your backend URL

### Gemini API Issues
- Verify your API key is set correctly
- Check API quota/limits
- Ensure internet connection is active

## Testing the API

### Test Prediction Endpoint:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"radius":1.0,"temp":288,"insol":1.0,"period":365,"steff":5778,"sradius":1.0}'
```

### Test Comparison Endpoint:
```bash
curl -X POST http://localhost:5000/api/compare \
  -H "Content-Type: application/json" \
  -d '{"source":"Earth","target":"Mars"}'
```

### Health Check:
```bash
curl http://localhost:5000/api/health
```
