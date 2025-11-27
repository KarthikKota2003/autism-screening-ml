# Quick Test Script - Verify Pipeline Dependencies

import sys

print("Checking required packages...")
print("="*60)

required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'imblearn': 'imbalanced-learn',
    'category_encoders': 'category-encoders',
    'skopt': 'scikit-optimize',
    'scipy': 'scipy'
}

missing_packages = []

for module, package in required_packages.items():
    try:
        __import__(module)
        print(f"[OK] {package:25s} - OK")
    except ImportError:
        print(f"[X] {package:25s} - MISSING")
        missing_packages.append(package)

print("="*60)

if missing_packages:
    print(f"\n[ERROR] Missing {len(missing_packages)} package(s):")
    for pkg in missing_packages:
        print(f"   - {pkg}")
    print(f"\nInstall with:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)
else:
    print("\n[SUCCESS] All required packages are installed!")
    print("\nPipeline is ready to run!")
    print("\nUsage:")
    print("  python ml_pipeline.py --dataset toddler --input Autism_Toddler_Data_Preprocessed.csv --output_dir results/toddler")
