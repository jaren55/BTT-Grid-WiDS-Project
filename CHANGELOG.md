# Changelog

All notable changes to this wildfire evacuation risk prediction project are documented here.

## [0.2.0] - 2026-03-10

### Added
- Modular project structure separating notebooks, source code, and results
- `src/features.py` module for wildfire feature engineering
- Feature generation function `make_features()`
- `results/` directory for model outputs and submission files
- `requirements.txt` for dependency management
- `.gitignore` for excluding unnecessary files

### Changed
- Refactored original notebook into multiple notebooks:
  - `01_data_preprocessing.ipynb`
  - `02_data_visualization.ipynb`
  - `03_model_training.ipynb`
  - `04_results_analysis.ipynb`
- Improved code readability and organization

---

## [0.1.0] - 2026-03-10

### Added
- Initial wildfire survival analysis notebook
- Dataset loading and exploratory data analysis
- Random Survival Forest model for predicting evacuation zone threats
- Evaluation metrics including Concordance Index and Brier Score
- Probability prediction at 12h, 24h, 48h, and 72h horizons