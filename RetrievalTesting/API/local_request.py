import requests, time

# testing api locally
# testing retrieval
BASE = "http://localhost:8000/v1/retrieval"

def post(payload, fmt="json", label=""):
    url = f"{BASE}?format={fmt}"
    r = requests.post(url, json=payload, timeout=300)
    print(f"\n[{label}] POST {url} -> {r.status_code}")
    try:
        data = r.json()
        print("Body:", data)
    except Exception:
        print("Body (raw):", r.text)
        return None, r
    return data, r

# just to check the whole thing's running
print("Health:", requests.get("http://localhost:8000/health", timeout=30).status_code)

# testing that using default weights works
payload1 = {
    "query": "got a copyright claim for a video critiquing music and playing for context. I am playing no more than ten to fifteen seconds of the clip, then pausing and discussing the musical elements, context behind the music video, and more. The song is 'umbrella' by rihanna. Do you think this falls under Fair Use of DMCA?",
    "top_k": 25
}
data1, r1 = post(payload1, "json", "default-weights")

# avoid hitting rate limit
time.sleep(5)

# testing normalization of weights that don't add to 1
payload2 = {
    "query": "got a copyright claim for a video critiquing music and playing for context. I am playing no more than ten to fifteen seconds of the clip, then pausing and discussing the musical elements, context behind the music video, and more. The song is 'umbrella' by rihanna. Do you think this falls under Fair Use of DMCA?",
    "top_k": 20,
    "weights": {"similarity": 1.2, "court_stats": 0.6, "citation": 0.2}
}
data2, r2 = post(payload2, "json", "custom-weights")
if r2.status_code == 200:
    print("Normalized weights:", data2["meta"]["weights"])

time.sleep(5)

# testing csv functionality

csv = requests.post(f"{BASE}?format=csv", json=payload2, timeout=300)
print("\n[csv] status:", csv.status_code)
if csv.status_code == 200:
    open("results.csv", "wb").write(csv.content)
    print("Saved results.csv")
else:
    print("CSV error:", csv.text)


# testing analysis
BASE = "http://localhost:8000/v1/analysis"
data1, r1 = post(payload1, "default-weights")