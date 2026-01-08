from flask import Flask
from web_app.routes import main
import os

app = Flask(__name__)
app.register_blueprint(main)

def test_export():
    client = app.test_client()
    
    # 1. Test missing file
    save_path = os.path.join(os.path.abspath("web_app"), '../instance/latest_results.csv')
    if os.path.exists(save_path):
        os.remove(save_path)
        
    resp = client.get('/download_results')
    print(f"Missing File Response: {resp.status_code}")
    
    # 2. Test existing file
    # Ensure instance dir exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("P_NAME,Score\nTest,99.9")
        
    resp = client.get('/download_results')
    print(f"Existing File Response: {resp.status_code}")
    print(f"Content Disposition: {resp.headers.get('Content-Disposition')}")
    print(f"Data: {resp.data.decode().strip()}")

if __name__ == "__main__":
    test_export()
