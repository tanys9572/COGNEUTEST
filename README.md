# COGNEUTEST
PROJECT TITLE: PREDICTIVE MODELS FOR COGNITIVE DECLINE ASSESSMENT

AUTHOR: TAN YAN SAN

## PHASE 1 Model Development

SETUP INSTRUCTIONS:
To run the code on VS code, please follow these steps:

1.Open your terminal (before that make sure have installed Python and Jupyter extension)

2.Create a virtual environment using the following command: 
python -m venv venv

3.Activate the virtual environment: 
venv\Scripts\activate

4.Install all libraries using the following command: 
pip install -r required_library.txt

5.After installing all libraries, open the project folder in VS Code.

6.Run the main Python script (Predictive Model Development.ipynb) for model training and evaluation

NOTE: 
- The Python version used in this study was Python 3.14.2.
- **NACC dataset is NOT allowed to be distributed to third party.**
If you need the dataset to run the script, please request one from https://nacc.redcap.rit.uw.edu/surveys/?s=KHNPKLJW8TKAD4DA
- The requested NACC dataset used in this study was frozen in September 2025.
- ALL ML algorithms took less than 30 min to run.
- The hyperparameter tuning took more than 30 min and SVM tuning took 2 hours 30 min to run.
- SHAP analysis took 2 hours to complete.

-----------------------------------------------------------------------------------------------------
## PHASE 2 System Development

1. Open your terminal 

   *Make sure virtual environment, venv is created

   *Make sure you have:
    - installed needed libraries to venv
    - .streamlit folder
    - logo dark.png, 
    - saved .pkl files after running phase 1 using Predictive Model Development.ipynb (imputer_value.pkl, minmax_scaler.pkl, onehot_encoder.pkl, selected_features.pkl & best_rf_model.pkl)

2. Use command to run the system: 
streamlit run system.py 

3. The system utilises a local SQLite for patient record management. You can test the system with the cogneutest.db generated within the directory the moment you launch the system for the first time.! The interface will open up inside your default web browser.
