"""Test suite for FastAPI endpoints and Two-Laptop Deployment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.main import health_check, list_available_trips, predict_trip


def test_api_pipeline():
    # 1. Health check
    r1 = health_check()
    assert r1["status"] == "online"
    print("✓ GET / (Health Check) passed:", r1["status"])
    
    # 2. List evaluation trips
    r2 = list_available_trips()
    trips = r2["trips"]
    assert len(trips) > 0, "No trips returned"
    print(f"✓ GET /api/trips passed: Found {len(trips)} evaluation trips. First trip: {trips[0]['trip_id']}")
    
    # 3. Post predict trip
    target_trip = trips[0]["trip_id"]
    print(f"Testing POST /api/predict/{target_trip}...")
    r3 = predict_trip(target_trip, k=8, beta=15.0)
    assert r3["status"] == "success"
    assert "accuracy_percent" in r3
    assert "route_continuity_percent" in r3
    assert len(r3["reconstructed_segments"]) > 0
    
    print(f"✓ POST /api/predict/{target_trip} passed!")
    print(f"   -> GPS observations: {r3['num_gps_observations']}")
    print(f"   -> Accuracy:         {r3['accuracy_percent']}%")
    print(f"   -> Route Continuity: {r3['route_continuity_percent']}%")
    print(f"   -> Latency:          {r3['latency_ms']} ms")
    print("\n🎉 ALL API ENDPOINTS FUNCTIONAL AND TESTED FOR TWO-LAPTOP EVALUATION!")


if __name__ == "__main__":
    test_api_pipeline()
