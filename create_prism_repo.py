
from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parent
REPO_NAME = "PRISM"
GH = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"

def run(command, cwd=ROOT):
    print("+", " ".join(map(str, command)))
    subprocess.run(command, cwd=cwd, check=True)

def write_file(relative_path, content):
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")

def main():
    if not Path(GH).exists():
        print("GitHub CLI not found:", GH)
        sys.exit(1)

    run(["git", "--version"])
    run([GH, "auth", "status"])

    visibility = input(
        "GitHub repository visibility (private/public) [private]: "
    ).strip().lower() or "private"

    if visibility not in {"private", "public"}:
        print("Choose private or public.")
        sys.exit(1)

    if not (ROOT / ".git").exists():
        if any(p.name != Path(__file__).name for p in ROOT.iterdir()):
            print("Unexpected existing files. Stopping to avoid overwriting.")
            sys.exit(1)
    else:
        print("Git repository already initialized.")

    directories = [
        "configs", "data/raw", "data/processed", "src/data",
        "src/baseline", "src/signals", "src/detection",
        "src/correction", "src/verification", "src/evaluation",
        "tests/data", "tests/baseline", "tests/signals",
        "tests/detection", "tests/correction", "tests/verification",
        "tests/evaluation", "scripts", "docs", "results/plots",
        "experiments",
    ]

    for directory in directories:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    write_file("README.md", """# PRISM
Prediction Reliability and Intelligent Self-Correction Mechanism

Pipeline: Detect -> Diagnose -> Correct -> Verify -> Accept / Correct / Abstain.

Initial benchmark: IPL second-innings outcome prediction.

## Setup
Install Python 3.11+, then run:
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt

See docs/experiment_protocol.md before experiments.
This repository is a research scaffold; placeholder modules need implementation.
""")

    write_file(".gitignore", """__pycache__/
*.py[cod]
.venv/
venv/
.env
.pytest_cache/
data/raw/*
data/processed/*
models/
checkpoints/
*.pkl
*.joblib
*.pt
*.pth
*.keras
results/*.parquet
results/*.csv
results/*.json
""")

    write_file("requirements.txt", """numpy
pandas
scikit-learn
scipy
PyYAML
joblib
matplotlib
pytest
""")

    for name in [
        "data", "architecture", "experiment_protocol",
        "feature_dictionary", "leakage_audit", "data_quality_report",
        "baseline_report", "signal_report", "detection_report",
        "similarity_correction_report", "dynamic_selection_report",
        "verification_report", "robustness_report",
        "evaluation_protocol", "claim_evidence_table",
    ]:
        write_file(
            f"docs/{name}.md",
            f"# {name.replace('_', ' ').title()}\n\nTODO: Document this component.\n",
        )

    write_file("data/README.md", """# Data
Download the source dataset separately. Do not commit large raw data.
Document source, license, download steps, and preprocessing here.
""")

    py_files = [
        "src/interfaces.py", "src/prism.py",
        "src/data/parse_cricsheet.py", "src/data/clean_matches.py",
        "src/data/schemas.py", "src/data/build_states.py",
        "src/data/build_features.py", "src/data/split_matches.py",
        "src/baseline/train.py", "src/baseline/predict.py",
        "src/baseline/metrics.py",
        "src/signals/confidence.py", "src/signals/entropy.py",
        "src/signals/calibration.py", "src/signals/similarity.py",
        "src/signals/extract.py",
        "src/detection/build_targets.py", "src/detection/train_detector.py",
        "src/detection/predict_risk.py", "src/detection/metrics.py",
        "src/correction/retrieval.py",
        "src/correction/similarity_corrector.py",
        "src/correction/candidate_schema.py",
        "src/correction/model_candidates.py",
        "src/correction/dynamic_selector.py",
        "src/correction/disagreement.py",
        "src/verification/evidence.py",
        "src/verification/decision_policy.py",
        "src/verification/thresholds.py",
        "src/verification/metrics.py",
        "src/evaluation/metrics.py",
        "src/evaluation/perturbations.py",
        "src/evaluation/temporal_shift.py",
        "src/evaluation/ood_analysis.py",
        "src/evaluation/ablation_runner.py",
        "src/evaluation/statistics.py",
        "scripts/run_prism.py", "scripts/run_robustness.py",
        "scripts/run_ablations.py",
    ]

    for name in py_files:
        write_file(
            name,
            '"""TODO: Implement and test this PRISM component."""\n\n'
            "def main():\n"
            "    raise NotImplementedError('Implementation pending')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
        )

    test_files = [
        "tests/test_pipeline.py",
        "tests/data/test_parser.py", "tests/data/test_cleaning.py",
        "tests/data/test_features.py", "tests/data/test_no_leakage.py",
        "tests/baseline/test_predictions.py",
        "tests/signals/test_signals.py",
        "tests/detection/test_detector.py",
        "tests/correction/test_retrieval.py",
        "tests/correction/test_similarity_corrector.py",
        "tests/correction/test_dynamic_selector.py",
        "tests/correction/test_disagreement.py",
        "tests/verification/test_decisions.py",
        "tests/evaluation/test_metrics.py",
        "tests/evaluation/test_perturbations.py",
    ]

    for name in test_files:
        write_file(name, '"""TODO: Add tests for this module."""\n')

    configs = {
        "configs/data.yaml": "dataset: ipl\nprediction_task: second_innings_winner\n",
        "configs/baseline.yaml": "random_seed: 42\nmodels: [logistic_regression, random_forest]\n",
        "configs/detection.yaml": "threshold_selection: validation_only\n",
        "configs/correction.yaml": "enable_similarity: true\nenable_dynamic_selection: true\n",
        "configs/evaluation.yaml": "report_coverage: true\nreport_harmful_corrections: true\n",
        "configs/pipeline.yaml": "decision_modes: [ACCEPT, CORRECT, ABSTAIN]\n",
        "configs/signals.yaml": "# TODO: Configure reliability signals\n",
        "configs/similarity_correction.yaml": "# TODO: Configure neighbor retrieval\n",
        "configs/dynamic_selection.yaml": "# TODO: Configure model selection\n",
        "configs/verification.yaml": "# TODO: Configure verification thresholds\n",
        "configs/robustness.yaml": "# TODO: Configure failure scenarios\n",
        "configs/splits.yaml": "# Keep all snapshots from a match in one split\n",
        "configs/ablation.yaml": "# TODO: Define ablation experiments\n",
    }

    for name, content in configs.items():
        write_file(name, content)

    write_file("results/.gitkeep", "")
    write_file("results/plots/.gitkeep", "")
    write_file("experiments/.gitkeep", "")

    # Initialize Git if necessary, then stage and commit the scaffold.
    if not (ROOT / ".git").exists():
        run(["git", "init", "-b", "main"])
    run(["git", "add", "."])

    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT
    )
    if status.returncode != 0:
        run(["git", "commit", "-m", "Create PRISM research scaffold"])
    else:
        print("No new changes to commit.")

    # If origin already exists, don't overwrite it.
    remotes = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT, capture_output=True, text=True,
    )

    if remotes.returncode == 0:
        print("Existing origin:", remotes.stdout.strip())
        answer = input("Use this existing origin? (yes/no): ").strip().lower()
        if answer != "yes":
            print("Stopped. Existing remote was left unchanged.")
            sys.exit(1)
        run(["git", "push", "-u", "origin", "main"])
    else:
        run([
            GH, "repo", "create", REPO_NAME, f"--{visibility}",
            "--source", str(ROOT), "--remote", "origin", "--push",
        ])

    print("\nPRISM scaffold published successfully.")
    print("Local directory:", ROOT)
    print("GitHub account: Swarup625")
    print("Open: https://github.com/Swarup625/PRISM")

if __name__ == "__main__":
    main()
