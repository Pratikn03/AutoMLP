import subprocess

# List of trainer scripts for boosters and AutoML frameworks
scripts = [
    "Project/trainers/train_boosters.py",
    "Project/trainers/train_catboost.py",
    "Project/trainers/train_flaml.py",
    "Project/trainers/train_h2o.py",
    # Add more scripts if needed, e.g., AutoGluon/LightAutoML
]

for script in scripts:
    print(f"Running: {script}")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
