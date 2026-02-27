# PolyLoop‑BO — Closed-loop electrospinning optimisation

**What it is:** a mini research platform that demonstrates how to combine:

- **Bayesian optimisation (BO)** for selecting the next experiments
- **Vision-based characterisation** (synthetic SEM-like images → fibre diameter distribution)
- **Text / literature-informed priors** (RAG-style retrieval → bounds and feasibility constraints)

This is intentionally written as a *deployable* starting point: you can swap the synthetic simulator
for your real electrospinning + imaging + assay pipelines.

---

## Why this is useful (research angle)

Electrospinning is a classic *high-dimensional* process where morphology and biofunction depend on
interacting variables (polymer blend, voltage, flow rate, humidity, additives).
The novelty here is an **end-to-end closed loop**:

1. BO proposes next formulation/conditions
2. an experiment is run (simulated here)
3. characterisation occurs automatically (image → diameter metrics)
4. results are appended and BO updates

The same pattern applies to **robotic chemistry** and other materials platforms.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Run demos

### 1) Closed-loop optimisation (simulated)
```bash
polyloop simulate --rounds 5 --seed 7
```

### 2) Fibre vision demo
```bash
polyloop vision-demo --n-images 6 --seed 7
```

---

## Data

- `data/electrospinning_experiments.csv` — synthetic but chemically plausible dataset
- `data/fiber_images/` — synthetic SEM-like images + `index.csv` mapping images to experiments

> Replace with your own data once you connect the workflow to a real lab.

---

## Key files

- `src/polyloop/bo.py` — acquisition + candidate selection
- `src/polyloop/models.py` — surrogate models (GP via scikit-learn) + uncertainty
- `src/polyloop/vision.py` — synthetic image generator + diameter estimator
- `src/polyloop/llm_retrieval.py` — TF‑IDF retrieval over a local “literature snippets” corpus
- `src/polyloop/simulator.py` — synthetic ground-truth mapping from parameters → outcomes

---

## Notes on extension

- Swap in **BoTorch** later for state-of-the-art Bayesian optimisation (qEI/qNEI/EHVI)
- Add **multi-fidelity**: cheap optical images for fast feedback + expensive SEM/assays for validation
- Add robotic adapters (e.g., Opentrons / custom lab scripts) in `src/polyloop/lab_adapters/`
