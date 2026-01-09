import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "P_MASS_EST": 1.1,
    "P_RADIUS_EST": 1.05,
    "P_PERIOD": 365,
    "P_TEMP_EQUIL": 288,
    "S_TEMPERATURE": 5778
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
