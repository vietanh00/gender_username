import requests
import json


headers = {'accept': 'application/json'}
url = "http://0.0.0.0:8000/predict"

js_data = {'name': "Trần Trọng Nguyên"}
response = requests.get(url, params=js_data, headers=headers)
print(response.json())