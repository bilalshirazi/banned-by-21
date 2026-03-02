import sys
import os
import glob
from PIL import Image

# Add the project root to the path so we can import app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import get_eligibility, analyze_image, RESTRICTED_JOBS
except ImportError as e:
    print(f"Error: Could not import app logic. {e}")
    sys.exit(1)

def run_asset_tests():
    image_paths = sorted(glob.glob("data/test_images/*.png"))
    
    if not image_paths:
        print("❌ No images found in data/test_images/")
        return

    print(f"🚀 Starting Verification of {len(image_paths)} assets...\n")
    print(f"{'IMAGE':<20} | {'EXPECTED':<12} | {'RESULT':<12} | {'AI LABEL':<30} | {'STATUS'}")
    print("-" * 100)

    pass_count = 0
    fail_count = 0

    # Test each image assuming a restricted role (Teacher)
    test_job = RESTRICTED_JOBS[0] 

    for path in image_paths:
        filename = os.path.basename(path)
        img = Image.open(path)
        
        # 1. Run AI Analysis
        label, confidence = analyze_image(img)
        
        # 2. Run Full Logic
        result_html = get_eligibility(img, test_job)
        
        # 3. Determine Expected Outcome based on filename
        is_baseline = any(x in filename for x in ["baseline", "medical-mask", "baseball-cap", "headphones"])
        expected = "Eligible" if is_baseline else "Ineligible"
        
        # Check actual result
        actual = "Ineligible" if "Ineligible" in result_html else "Eligible"
        
        # Calculate Pass/Fail
        status = "✅ PASS" if expected == actual else "❌ FAIL"
        
        if status == "✅ PASS": pass_count += 1
        else: fail_count += 1

        print(f"{filename:<20} | {expected:<12} | {actual:<12} | {label:<30} | {status}")

    print("-" * 100)
    print(f"\n📊 SUMMARY: {pass_count} Passed, {fail_count} Failed.")
    
    if fail_count > 0:
        print("\n⚠️  Action Required: Some images were misidentified. You may need to refine the DETECTION_LABELS or thresholds in app.py.")
    else:
        print("\n✨ Perfect Accuracy: Your dataset is fully compatible with the current AI calibration.")

if __name__ == "__main__":
    run_asset_tests()
