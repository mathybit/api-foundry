import base64
from io import BytesIO
import json
import numpy as np
import requests


##### API URLs
# Local PC
BASE_URL = r'http://127.0.0.1:5000'

# Production - Windows machine
#BASE_URL = r'http://10.0.0.19:80'

# Production - Linux machine
#BASE_URL = r'http://10.0.0.5:80'


##### API endpoints
# API general endpoints
API_VERSION_ENDPOINT = '/version'
API_SERVICES_ENDPOINT = '/services'

# Service endpoints
VERSION_ENDPOINT = '/service/example_model/version'
SCHEMA_ENDPOINT = '/service/example_model/schema'
COMPUTE_ENDPOINT = '/service/example_model/predict'


##### Call the API
# Call API endpoints
print('=' * 80)
print('GET:', API_VERSION_ENDPOINT)
response = requests.get(BASE_URL + API_VERSION_ENDPOINT, verify=False)
print('Status:', response.status_code, 'Reason:', response.reason)
print('Response:', response.text)
resdict = response.json()
print('JSON:\n', json.dumps(resdict, indent=2))


print('=' * 80)
print('GET:', API_SERVICES_ENDPOINT)
response = requests.get(BASE_URL + API_SERVICES_ENDPOINT, verify=False)
print('Status:', response.status_code, 'Reason:', response.reason)
print('Response:', response.text)
resdict = response.json()
print('JSON:\n', json.dumps(resdict, indent=2))


# Call service endpoints
print('\n' + '=' * 80)
print('GET:', VERSION_ENDPOINT)
response = requests.get(BASE_URL + VERSION_ENDPOINT, verify=False)
print('Status:', response.status_code, 'Reason:', response.reason)
print('Response:', response.text)
resdict = response.json()
print('JSON:\n', json.dumps(resdict, indent=2))


print('\n' + '=' * 80)
print('GET:', SCHEMA_ENDPOINT)
response = requests.get(BASE_URL + SCHEMA_ENDPOINT, verify=False)
print('Status:', response.status_code, 'Reason:', response.reason)
print('Response:', response.text)
resdict = response.json()
print('JSON:\n', json.dumps(resdict, indent=2))


print('\n' + '=' * 80)
print('POST:', COMPUTE_ENDPOINT)
payload = {
    'input': np.random.randint(0, int(1e9)),
    'request_id': np.random.randint(0, int(1e9))
}
print('Payload:', json.dumps(payload, indent=2))

response = requests.post(BASE_URL + COMPUTE_ENDPOINT, json=payload, verify=False)
print('Status:', response.status_code, 'Reason:', response.reason)
print('Response:', response.text)
resdict = response.json()
print('JSON:\n', json.dumps(resdict, indent=2))
