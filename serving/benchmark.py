import time
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict"

SAMPLE_PAYLOAD = {
    "tenure": 12,
    "MonthlyCharges": 75.50,
    "TotalCharges": 906.00,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def run_benchmark(n_requests: int = 500):
    # Warm-up requests
    for _ in range(10):
        requests.post(API_URL, json=SAMPLE_PAYLOAD)

    latencies_ms = []
    start_total = time.perf_counter()

    for _ in range(n_requests):
        t0 = time.perf_counter()
        resp = requests.post(API_URL, json=SAMPLE_PAYLOAD)
        t1 = time.perf_counter()
        assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
        latencies_ms.append((t1 - t0) * 1000.0)

    total_time = time.perf_counter() - start_total
    latencies = np.array(latencies_ms)

    print(f"=== API Latency Benchmark ({n_requests} requests) ===")
    print(f"Throughput: {n_requests / total_time:.2f} req/sec")
    print(f"Mean:       {np.mean(latencies):.2f} ms")
    print(f"p50:        {np.percentile(latencies, 50):.2f} ms")
    print(f"p95:        {np.percentile(latencies, 95):.2f} ms")
    print(f"p99:        {np.percentile(latencies, 99):.2f} ms")


if __name__ == "__main__":
    run_benchmark(500)
