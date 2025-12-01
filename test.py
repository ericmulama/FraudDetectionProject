import sys

libraries = [
    "os",
    "gc",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "lightgbm",
    "shap",
]

print("--- Checking installed libraries in current environment ---")
print(f"Python Executable: {sys.executable}\n")

for lib in libraries:
    try:
        __import__(lib)
        print(f"✅ SUCCESS: '{lib}' is installed.")
    except ImportError:
        print(f"❌ MISSING: '{lib}' is NOT installed. You need to install it.")
    except Exception as e:
        print(f"⚠️ ERROR: An unexpected error occurred while importing '{lib}': {e}")

print("\n--- Check complete ---")
