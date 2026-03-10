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

## Installation
