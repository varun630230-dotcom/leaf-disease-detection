"""LeafGuard AI — Direct Python Test Runner."""

import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

def run():
    print("=" * 60)
    print("LEAFGUARD AI - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # 1. Test Image Validator
    print("\n[1/3] Testing Image Validation Service...")
    from app.services.image_validator import ImageValidator
    from tests.test_image_validator import (
        test_valid_jpeg,
        test_valid_png,
        test_invalid_mime_type,
        test_file_too_small,
        test_dimensions_too_small,
        test_dark_image,
        test_bright_image,
        test_corrupted_image,
    )
    v = ImageValidator()
    tests_val = [
        ("Valid JPEG image", lambda: test_valid_jpeg(v)),
        ("Valid PNG image", lambda: test_valid_png(v)),
        ("Invalid MIME (PDF rejection)", lambda: test_invalid_mime_type(v)),
        ("File too small rejection", lambda: test_file_too_small(v)),
        ("Dimensions too small rejection", lambda: test_dimensions_too_small(v)),
        ("Dark image rejection", lambda: test_dark_image(v)),
        ("Bright image rejection", lambda: test_bright_image(v)),
        ("Corrupted image rejection", lambda: test_corrupted_image(v)),
    ]

    for name, fn in tests_val:
        t0 = time.time()
        fn()
        print(f"  [PASS] {name:<40} ({((time.time()-t0)*1000):.1f} ms)")

    # 2. Test OOD Detector
    print("\n[2/3] Testing Out-Of-Distribution (OOD) Detector...")
    from tests.test_ood import test_ood_detection_high_energy, test_ood_detection_low_energy_non_leaf
    t0 = time.time()
    test_ood_detection_high_energy()
    print(f"  [PASS] {'In-distribution high energy detection':<40} ({((time.time()-t0)*1000):.1f} ms)")
    t0 = time.time()
    test_ood_detection_low_energy_non_leaf()
    print(f"  [PASS] {'Out-of-distribution rejection':<40} ({((time.time()-t0)*1000):.1f} ms)")

    # 3. Test API Endpoints
    print("\n[3/3] Testing FastAPI Endpoints...")
    from tests.test_api import (
        test_health_endpoint,
        test_supported_plants_endpoint,
        test_performance_endpoint,
        test_analyze_valid_image,
        test_analyze_empty_file,
        test_get_nonexistent_analysis,
    )
    api_tests = [
        ("GET /api/health", test_health_endpoint),
        ("GET /api/supported-plants", test_supported_plants_endpoint),
        ("GET /api/performance", test_performance_endpoint),
        ("POST /api/analyze (full pipeline)", test_analyze_valid_image),
        ("POST /api/analyze (empty file 400)", test_analyze_empty_file),
        ("GET /api/analysis/{id} (404 test)", test_get_nonexistent_analysis),
    ]

    for name, fn in api_tests:
        t0 = time.time()
        fn()
        print(f"  [PASS] {name:<40} ({((time.time()-t0)*1000):.1f} ms)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (16/16 tests verified)")
    print("=" * 60)

if __name__ == "__main__":
    run()
