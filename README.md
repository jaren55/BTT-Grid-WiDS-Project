# BTT-Grid-WiDS-Project 
This project builds a survival analysis–based forecasting system that estimates the probability a wildfire will threaten evacuation zones within 12, 24, 48, and 72 hours, using only the first 5 hours of observed fire behavior. By producing calibrated, time-dependent risk forecasts, the system supports time-sensitive decision-making under uncertainty, aligning with real-world emergency management needs.

## Project Description
This project: 
- Predicts the probability that an active wildfire will threaten nearby evacuation zones within 12, 24, 48, and 72 hours.
- Uses survival analysis models to estimate time-to-threat.
- Handles right-censored wildfire outcomes, where a fire may not ultimately impact a given zone.
- Evaluates model perfermance using C-index to measure how well the system ranks urgent wildfire threats and Brier score to assess how accurately predicted probabilities reflect real-world outcomes.

## Technologies Used
- Python
- pandas / numpy (data processing)
- survival analysis libraries (scikit-survival/lifelines)
- matplotlib or seaborn for visualization
- Git and GitHub for version control

## File Structure 
wids-wildfire-survival/
│
├── data/                      
│   ├── train.csv
│   └── test.csv
│
├── notebooks/                
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_data_visualization.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── src/                      
│   ├── __init__.py
│   └── features.py            
│
├── results/                  
│   └── submission.csv
│
├── test_cases/               
│   ├── test_case_1.md
│   ├── test_case_2.md
│   └── test_case_3.md
│
├── requirements.txt           
├── README.md                 
├── CHANGELOG.md              
└── .gitignore      
           
## Installation

Clone the repository:

git clone https://github.com/jaren55/BTT-Grid-WiDS-Project.git
cd BTT-Grid-WiDS-Project

Install dependencies:

pip install -r requirements.txt

## Running the Project

Launch Jupyter Notebook:

jupyter notebook

Run the notebooks in order:

1. notebooks/01_data_preprocessing.ipynb
2. notebooks/02_data_visualization.ipynb
3. notebooks/03_model_training.ipynb
4. notebooks/04_results_analysis.ipynb