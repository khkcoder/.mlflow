import requests
import json

# --- Configuration ---

# This is the endpoint for your Docker container.
# If you are testing the local 'mlflow models serve' (non-Docker)
# deployment, change the port from 5003 to 5001.
MODEL_API_URL = "http://127.0.0.1:5001/invocations"

# This payload uses the 'dataframe_split' format, which includes
# the column names. This is the most robust way to send data.
test_data = {
    "dataframe_split": {
        "columns": [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol"
        ],
        "data": [
            # You can add more rows here to test batch predictions
            [7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8],
            [6.3, 0.3, 0.34, 1.6, 0.049, 14, 132, 0.994, 3.3, 0.49, 9.5]
        ]
    }
}

print(f"🚀 Sending test request to: {MODEL_API_URL}")

try:
    # Send the POST request
    # The 'json' parameter automatically serializes the dict
    # and sets the 'Content-Type: application/json' header.
    response = requests.post(MODEL_API_URL, json=test_data)

    # Raise an exception if the server returned an error (e.g., 404, 500)
    response.raise_for_status()

    # If we're here, the request was successful (200 OK)
    predictions = response.json()
    
    print("\n✅ Success! Model server responded.")
    print("---------------------------------")
    print("Predictions:")
    print(json.dumps(predictions, indent=2))
    print("---------------------------------")

except requests.exceptions.ConnectionError:
    print(f"\n❌ Error: Could not connect to the model server at {MODEL_API_URL}.")
    print("Please make sure your model server or Docker container is running.")

except requests.exceptions.HTTPError as http_err:
    print(f"\n❌ HTTP Error occurred: {http_err}")
    print(f"Response content: {response.text}")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")