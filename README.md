# Object-Oriented Programming Mini-Projects

This repository contains five educational, class-based Python mini-projects. Each project keeps reusable models in `src/` and presents its analysis in a Jupyter notebook. Numerical inputs and outputs are for learning and demonstration; check the assumptions and consult authoritative data before using results for real-world decisions.

## Projects

| Project | Reusable module | Analysis notebook | Description |
|---|---|---|---|
| UBOS district population | `src/ubos.py` | `ubos_analysis.ipynb` | Population summaries, forecasting models, validation and classroom-demand estimates. |
| Solar micro-grid dispatch | `src/solar.py` | `solar_dispatch.ipynb` | Solar/battery and hybrid dispatch, demand inputs, costs, feasibility and sensitivity analysis. The seeded sample demands are in `data/solar_demands_30_days.csv`. |
| Lake Victoria fish stock and export risk | `src/fish_stock.py` | `fish_stock_risk.ipynb` | Logistic fish-stock simulation, price/revenue risk, harvest scenarios and closed-season comparison. |
| Rainfall and crop suitability | `src/rainfall.py` | `rainfall_crop_suitability.ipynb` | Rainfall summaries, crop screening, regional similarity, season peaks and an illustrative advisory. |
| Taxi route revenue and fleet planning | `src/taxi.py` | `taxi_route_planner.ipynb` | Route revenue, equilibrium, walk-forward forecasts, fleet sizing and a weekly-demand extension. |

`tests/` contains pytest unit tests for the project classes and calculations. `.github/workflows/tests.yml` runs the test suite on pushes and pull requests.

## Setup

Python 3.12 or newer is recommended. From the repository root, create and activate a virtual environment, then install the development dependencies:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Open a notebook in VS Code or another Jupyter-compatible editor and select the environment where the requirements were installed as its Python kernel. Run notebook cells in order. Some notebooks include plots and explanatory conclusions; the results depend on their displayed data and assumptions.

## Run the tests

```bash
python -m pytest -q
```

To run tests for an individual project, for example:

```bash
python -m pytest tests/test_taxi.py -q
```

The test and runtime dependencies are declared in `requirements-dev.txt` and `requirements.txt`, respectively.

## Data and interpretation

Several project inputs are illustrative or synthetic and are labelled as such in their notebooks. In particular, the rainfall notebook uses the one-year monthly values supplied in the assignment rather than multi-year observations, and the taxi extension generates seeded synthetic demand. Forecasts, crop screens, fleet estimates, prices, costs and risk classifications demonstrate the specified methods; they are not official recommendations or guarantees. Review each notebook's assumptions, units and references before interpreting its outputs.

## Academic integrity and AI use

AI coding assistance was used during development to help draft and revise project classes, notebook analyses and explanations, pytest tests, and the test workflow; to investigate errors; and to validate the work by running tests and notebook code cells. AI assistance does not replace the student's responsibility for the submitted work.

