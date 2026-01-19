# Frontend (Streamlit)

This Streamlit app provides a minimal UI to predict exoplanet habitability and visualize model outputs.

## Run locally
1. Install dependencies:

   pip install -r frontend/requirements.txt

2. Start the backend API (in `notebook/api`):

   python app.py

3. Run the Streamlit app:

   streamlit run frontend/streamlit_app.py

Default API base URL is `http://localhost:5000`. You can change it in the sidebar.

---

## Quick test helpers

I added small helper scripts to simplify testing the API:

- `scripts/test_predict.py` — simple Python script that POSTs a sample payload to `/predict` and prints the response.

  Usage:

  ```bash
  python scripts/test_predict.py --url http://127.0.0.1:5000
  ```

- `scripts/test_predict.ps1` — PowerShell script using `Invoke-RestMethod` for Windows users.

  Usage:

  ```powershell
  .\scripts\test_predict.ps1 -Url 'http://127.0.0.1:5000'
  ```

- `tests/test_api.py` — pytest tests (requires server running at `http://127.0.0.1:5000`). Run `pytest -q` to execute.

If you prefer, I can run the pytest file locally and paste the results; confirm and I'll run it for you.