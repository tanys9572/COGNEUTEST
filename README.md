# COGNEUTEST
PROJECT TITLE: PREDICTIVE MODELS FOR COGNITIVE DECLINE ASSESSMENT

COGNEUTEST is a fast, transparent and affordable clinical support system designed for early screening of cognitive decline. It utilised a trained machine learning ensemble algorithm, Random Forest, to help the system to evaluate patient demographics, medical conditions, lifestyle, and influential cognitive variables to predict three cognitive statuses, which are Normal Cognition, Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD) dementia. The system incorporates Explainable AI SHapley Additive exPlanations (SHAP) to provide clinicians with features’ contribution levels behind every prediction results. 

-----------------------------------------------------------------------------------------------------
## PHASE 1 Model Development

SETUP INSTRUCTIONS:
To run the code on VS code, please follow these steps:

1.Open your terminal (before that make sure have installed Python and Jupyter extension)

2.Create a virtual environment using the following command: 
_python -m venv venv_

3.Activate the virtual environment: 
_venv\Scripts\activate_

4.Install all libraries using the following command: 
_pip install -r required_library.txt_

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
      _*Note: Due to file size limitations on GitHub, large model files such as best_rf_model.pkl are not committed directly to the repository. Alternatively, you can generate all five .pkl artifacts directly from scratch by running sections 1, 2, 3, 4, 5a, and 7 inside the provided Predictive Model Development.ipynb notebook to generate them. 
_
2. Use command to run the system: 
_streamlit run system.py _

3. The system utilises a local SQLite for patient record management. You can test the system with the cogneutest.db generated within the directory the moment you launch the system for the first time. Upon launching the system on browser, you are presented with the Risk Assessment page. There is no need for login or registration because COGNEUTEST is designed as a direct-access clinical decision support tool.
