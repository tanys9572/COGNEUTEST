import sqlite3
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import uuid
from datetime import datetime
import pytz
import plotly.express as px
import pandas as pd
import numpy as np
import shap
from io import BytesIO #for pdf
from xhtml2pdf import pisa #for pdf
import base64 #for pdf
import textwrap

st.set_page_config(page_title="COGNEUTEST", layout="wide", initial_sidebar_state="expanded")

current_year = datetime.now().year

#set pages
pages = ["Risk Assessment", "Dashboard", "Record Search", "About"]

#start session
if "current_page" not in st.session_state:
    query_page = st.query_params.get("page", "Risk Assessment")
    if query_page in pages:
        st.session_state.current_page = query_page
    else:
        st.session_state.current_page = "Risk Assessment"

#create session id
if "current_ref_id" not in st.session_state:
    unique_suffix = str(uuid.uuid4()).split("-")[0].upper()
    current_year = datetime.now().year
    st.session_state.current_ref_id = f"REF-{current_year}-{unique_suffix}"

#check database
try:
    db_connection = sqlite3.connect("cogneutest.db", check_same_thread=False)
    db_cursor = db_connection.cursor()
    
    #if not exist
    db_cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            NACCID TEXT,
            CDRSUM REAL,
            NACCMOCA REAL,
            TRAILB REAL,
            REMDATES REAL,
            TRAILA REAL,
            TAXES REAL,
            TRAVEL REAL,
            NACCBMI REAL,
            BILLS REAL,
            NACCAGE INTEGER,
            PAYATTN REAL,
            SHOPPING REAL,
            ALCFREQ REAL,
            EVENTS REAL,
            NACCGDS REAL,
            EDUC REAL,
            MEALPREP REAL,
            SMOKYRS REAL,
            MARISTAT REAL,
            GAMES REAL,
            RACE REAL,
            NACCNE4S REAL,
            SEX INTEGER,
            HYPERT INTEGER,
            DIABET REAL,
            MINTTOTS REAL,
            STATUS INTEGER,
            REFERENCE_ID TEXT PRIMARY KEY,
            REPORT_PDF BLOB COLLATE BINARY
        )
    """)
    db_connection.commit()


    #if connected
    status_color = "#15877B"  
    status_text = "Database Connected"

except Exception as conn_error:
    #if not
    status_color = "#FF7272"
    status_text = "Database Disconnected"
    db_cursor = None
    db_connection = None

#load pkl files first
@st.cache_resource
def load_machine_learning_assets():
    # Load everything safely using joblib directly from the file paths
    imputer = joblib.load("imputer_value.pkl")
    scaler = joblib.load("minmax_scaler.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    feature_list = joblib.load("selected_features.pkl")
    model = joblib.load("best_rf_model.pkl")
        
    # If selected_features.pkl was saved as a DataFrame, extract its column names list
    if hasattr(feature_list, "columns"):
        feature_list = list(feature_list.columns)
        
    return imputer, scaler, encoder, feature_list, model
#check
try:
    saved_imputer, saved_scaler, encoder, training_feature_order, best_rf_model = load_machine_learning_assets()
except FileNotFoundError as e:
    st.error(f"❌ Error: Could not locate files. {e}")

#function for changing pages
def change_page(page_name):
    st.session_state.current_page = page_name

#function to convert html to pdf
def convert_html_to_pdf(html_string):
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html_string, dest=pdf_buffer)
    return pdf_buffer.getvalue()


#for form submit dialog if empty field
@st.dialog("⚠️ Incomplete Field Detected")
def show_accuracy_warning_modal(fields_list):
    st.write(
        f"The following field(s) were left unentered or marked as unknown: "
        f"**{', '.join(fields_list)}**."
    )
    st.error(
        "**Notice:** Leaving fields blank forces the processing engine "
        "to run an unaligned feature matrix, which **will reduce final prediction accuracy**."
    )
    st.write("Are you sure you want to proceed?")
    
    #2 button
    modal_col1, modal_col2 = st.columns(2)
    with modal_col1:
        if st.button("❌ Cancel & Fix Inputs", use_container_width=True):
            st.rerun() #stop submit form
            
    with modal_col2:
        #continue to prediction
        if st.button(":material/play_arrow: Proceed Anyway", type="primary", use_container_width=True):
            st.session_state.bypass_approved = True
            st.rerun()

#for css adjust
st.markdown(
     """
    <style>
    
    /* for header adjust */
    h1, h2, h3 {
        font-weight: 700 !important;
    }
            
    [data-testid="stSidebar"]{
        width: 280px !important;
    }
    [data-testid="stSidebarContent"] {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    [data-testid="stSidebar"] div.stButton > button > div {
        justify-content: flex-start !important;
        width: 100%;
        padding-top:10px;
        padding-bottom:10px;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]{
        background-color: white !important;
        border-radius: 6px;
    }

    .main-header {
        font-size:26px !important;
    }

    @media (min-width: 1000px) {
        [data-testid="stSidebar"]{
            width: 280px !important;
        }
    }
    
    
    /* for line */
    hr {
        border: none !important;
        border-top: 2px solid #555555 !important;
        margin: 10px 0 10px 0 !important;
    }

    /* for dashboard card */
    .dash-card {
        background-color: white;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        overflow: hidden;
        margin-bottom: 20px;
    }
    .dash-card-header {
        background-color: #2D323E; 
        color: white;
        padding: 8px 15px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: flex;
    }
    .dash-card-body {
        padding: 15px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .dash-value {
        font-size: 32px;
        font-family: 'Oswald', sans-serif;
        font-weight: 500;
        color: #333;
        margin: 0;
    }
    .dash-subtext {
        font-size: 13px;
        color: #878787;
        margin-top: 5px;
    }
    

    /*dowanload buttom*/
    div.stDownloadButton > button {
        background-color: #15877B !important;
        color: white !important;               
        border: 1px solid #15877B !important;  
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #106b61 !important;
        border-color: #106b61 !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True
)
# for icon link
st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">',
    unsafe_allow_html=True
)
    
#sidemenu
with st.sidebar:
    #logo
    st.image("logo dark.png", use_container_width=True)
    
    #click to change page just like normal website
    selected_page = option_menu(
        menu_title=None,
        options=pages, 
        icons=["clipboard2-pulse", "bar-chart", "journal-medical", "info-circle"], #bootstrap icon names
        default_index= pages.index(st.session_state.current_page),
        styles={
            "container": {"margin": "0px!important", 
                          "padding": "0px!important", 
                          "background-color": "transparent"},
            "icon": {"color": "white", "font-size": "20px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "color": "white",
                "padding": "15px 10px",
                "--hover-color": "rgba(255, 255, 255, 0.1)" 
            },
            "nav-link-selected": {"background-color": "#15877B", 
                                  "font-weight": "600",}, #when click change colour
        }
    )


#to make sure page update when changed
if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.query_params["page"] = selected_page
    st.rerun()

#header
top_col1, top_col2 = st.columns([6.5, 3.5], vertical_alignment="center")

with top_col1:        
    st.markdown("""
        <div style="padding: 10px 0px;">
            <div style="font-size: 28px; font-weight: 800; color: #15877B;">
                COGNEUTEST <span style="font-size: 28px;border-left: 3px solid #15877B;margin-left: 8px;padding-left: 8px;font-weight: 600;">
                Cognitive Decline Assessment System</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

with top_col2:
    # system time calculations
    my_tz = pytz.timezone("Asia/Kuala_Lumpur")
    now_my = datetime.now(my_tz)
    malaysia_str = now_my.strftime("%d %b %Y")

    #html
    st.markdown(f"""
        <div style="text-align: right; line-height: 1.5; padding-right: 10px;">
            <div style="font-size: 17px; font-weight: 700; color: #15877B; letter-spacing: 0.3px;">
                <span style="color: {status_color}; margin-right: 5px;">●</span> {status_text}
            </div>
            <div style="font-size: 14px; color: #717171; font-weight: 600; margin-top: 1px; letter-spacing: 0.3px;">
                {malaysia_str} <span style="color: #717171; margin-right: 5px;"> MYT</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
st.divider()

# main content
if st.session_state.current_page == "Risk Assessment":
    st.markdown(f'<h2 class="main-header">{st.session_state.current_page}</h2>', unsafe_allow_html=True)
    st.markdown("Fill out the patient clinical information below to predict the cognitive status.")
    st.markdown("""
        <style>
        /*input test colour*/
        div[data-testid="stForm"] input {
            -webkit-text-fill-color: #2D323E !important;
            font-weight: 500 !important;
        }
        div[data-testid="stForm"] div[data-baseweb="select"] div {
            color: #2D323E !important;
            font-weight: 500 !important;
        }
        div[data-testid="stForm"] label p {
            font-size: 15px !important;
            padding: 2px 5px;
            font-weight: 600;
        }
        /*placeholder */
        div[data-baseweb="input"] input::placeholder {
            -webkit-text-fill-color:rgba(49, 51, 63, 0.4) !important;
            font-style: italic !important;
        }
        </style>
    """, unsafe_allow_html=True)


    if db_connection is not None:
        try:
            # Run the query out in the open so it loads on page boot
            existing_ids_df = pd.read_sql_query("SELECT DISTINCT NACCID FROM patient_records", db_connection)
            existing_ids = existing_ids_df["NACCID"].tolist()
        except Exception:
            existing_ids = []

    st.markdown(
        '##### <i class="bi bi-calendar-event" style="-webkit-text-stroke: 1px;margin-right: 5px; color: #15877B;"></i> <span style="color: #15877B;">Patient Visit Mode</span>', 
        unsafe_allow_html=True
    )
    visit_mode = st.radio(
        "Select Visit Type:",
        ["New Patient (Auto-Generate ID)", "Existing Patient (Follow-up Visit)"],
        horizontal=True
    )

    if visit_mode == "Existing Patient (Follow-up Visit)":

        if existing_ids:
            patient_master_id = st.selectbox(
                ":material/search: Search & Select Existing ID:",
                options=existing_ids,
                help="Type or click to choose a patient profile."
            )
            #fetch the most recent record for a patient
            try:
                query = f"SELECT * FROM patient_records WHERE NACCID = '{patient_master_id}' ORDER BY rowid DESC LIMIT 1"
                history_df = pd.read_sql_query(query, db_connection)
                
                if not history_df.empty:
                    latest_record = history_df.iloc[0]
                    #map from num to text
                    st.session_state["autofill_age"] = int(latest_record.get("NACCAGE", 70))
                    db_educ = latest_record.get("EDUC")
                    st.session_state["autofill_educ"] = int(db_educ) if not pd.isna(db_educ) else None
                    st.session_state["autofill_smokyrs"] = int(latest_record.get("SMOKYRS")) if not pd.isna(latest_record.get("SMOKYRS")) else None
                    db_sex = latest_record.get("SEX")
                    if pd.isna(db_sex):
                        st.session_state["autofill_sex"] = "Unknown"
                    else:
                        st.session_state["autofill_sex"] = "Female" if int(db_sex) == 2 else "Male"
                    db_hypert = latest_record.get("HYPERT")
                    st.session_state["autofill_hypert"] = "Yes" if db_hypert == 1.0 else ("No" if db_hypert == 0.0 else "Unknown")
                    db_diabet = latest_record.get("DIABET")
                    diabet_map = {0.0: "No", 1.0: "Yes, Type I", 2.0: "Yes, Type II", 3.0: "Yes, other type"}
                    st.session_state["autofill_diabet"] = diabet_map.get(db_diabet, "Unknown") if not pd.isna(db_diabet) else "Unknown"
                    db_alcfreq = latest_record.get("ALCFREQ")
                    alc_map = {
                        0.0: "None/Less than once a month",
                        1.0: "About once a MONTH",
                        2.0: "About once a WEEK",
                        3.0: "A few times a week",
                        4.0: "Daily or almost daily"
                    }
                    st.session_state["autofill_alcfreq"] = alc_map.get(db_alcfreq, "Unknown") if not pd.isna(db_alcfreq) else "Unknown"
                    db_race = int(latest_record.get("RACE"))
                    race_map = {
                        1.0: "White", 2.0: "Black/African American", 3.0: "American Indian/Alaska Native",
                        4.0: "Native Hawaiian/Other Pacific Islander", 5.0: "Asian", 50.0: "Other/Multiracial"
                    }
                    st.session_state["autofill_race"] = race_map.get(db_race, "Unknown") if not pd.isna(db_race) else "Unknown"
                    
                    db_maristat = latest_record.get("MARISTAT")
                    mari_map = {
                        1.0: "Married", 2.0: "Widowed", 3.0: "Divorced", 
                        4.0: "Separated", 5.0: "Never Married", 6.0: "Living as married/domestic partner"
                    }
                    st.session_state["autofill_maristat"] = mari_map.get(db_maristat, "Unknown") if not pd.isna(db_maristat) else "Unknown"
                    
                    db_ne4s = latest_record.get("NACCNE4S")
                    apoe_map = {0.0: "No", 1.0: "1 copy", 2.0: "2 copies"}
                    st.session_state["autofill_ne4s"] = apoe_map.get(db_ne4s, "Unknown") if not pd.isna(db_ne4s) else "Unknown"
                    

                else:
                    st.session_state.clear() 
            except Exception as e:
                st.error(f"Error fetching historical record: {e}")
        else:
            st.warning("⚠️ No historical patient records found in the local database yet.")

    else:
        autofill_keys = [
            "autofill_age", "autofill_educ", "autofill_bmi", "autofill_smokyrs",
            "autofill_sex", "autofill_hypert", "autofill_diabet", "autofill_alcfreq",
            "autofill_race", "autofill_maristat", "autofill_ne4s"
        ]
        for key in autofill_keys:
            if key in st.session_state:
                del st.session_state[key]

        if "new_patient_token" not in st.session_state:
            unique_suffix = str(uuid.uuid4()).split('-')[0].upper()
            st.session_state.new_patient_token = f"PT-{current_year}-{unique_suffix}"
        
        patient_master_id = st.session_state.new_patient_token
        st.info(f"**New Patient Mode:** Current assessment will perform under Subject ID: `{patient_master_id}`")

    with st.form("assessment_form", enter_to_submit=False):
        st.markdown("""
            <div style="background-color: rgba(21, 135, 123, 0.03); border-left: 4px solid #15877B; padding: 12px; margin-bottom: 15px; border-radius: 4px;">
                <p style="margin: 0; font-size: 15px; color: #2D323E; font-weight: 500;">
                    <span style="color: red; font-weight: bold; font-size:18px;">*</span> Indicates a <strong style="font-weight: bold;">required field</strong>. 
                    Leaving fields blank or selecting "Unknown" will force the system to use default values, which will<strong style="font-weight: bold;"> reduce final prediction accuracy</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)
        #Demographics part
        st.markdown("""
            <div class="dash-card" style="margin-bottom: 5px;">
                <div class="dash-card-header">1. Patient Demographics</div>
            </div>
        """, unsafe_allow_html=True)
        
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        with demo_col1:
            naccage = st.number_input("Age :red[*]", min_value=18, max_value=130, value=st.session_state.get("autofill_age", 70), help="Patient age at visit. Acceptable range: 18 to 130")
            sex_options = ["Male", "Female"]
            sex_default = st.session_state.get("autofill_sex", "Male")
            sex_idx = sex_options.index(sex_default) if sex_default in sex_options else 0
            sex = st.selectbox("Sex :red[*]", options=sex_options, index=sex_idx, help="Biological sex")
        with demo_col2:
            race_options = ["White", "Black/African American", "Asian", "American Indian/Alaska Native", "Native Hawaiian/Other Pacific Islander", "Other/Multiracial", "Unknown"]
            race_default = st.session_state.get("autofill_race", "Unknown")
            race_idx = race_options.index(race_default) if race_default in race_options else 6
            race = st.selectbox("Race", options=race_options, index=race_idx, help="Primary demographic racial category")
            
            mari_options = ["Married", "Widowed", "Divorced", "Separated", "Never Married", "Living as married/domestic partner", "Unknown"]
            mari_default = st.session_state.get("autofill_maristat", "Unknown")
            mari_idx = mari_options.index(mari_default) if mari_default in mari_options else 6
            maristat = st.selectbox("Marital Status", options=mari_options, index=mari_idx, help="Current marital status")
        with demo_col3:
            educ = st.number_input(
                "Education Years", min_value=0, max_value=30, 
                value=st.session_state.get("autofill_educ", None), 
                placeholder="e.g. 12",
                help="Total formal education history in years. Range: 0 to 30"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        #lifestyle & medical history
        st.markdown("""
            <div class="dash-card" style="margin-bottom: 5px;">
                <div class="dash-card-header">2. Medical & Lifestyle History</div>
            </div>
        """, unsafe_allow_html=True)
        
        med_col1, med_col2, med_col3 = st.columns(3)
        with med_col1:
            naccbmi = st.number_input(
                "Body Mass Index (BMI)", min_value=10.0, max_value=100.0, step=0.1, format="%.1f",
                value=st.session_state.get("autofill_bmi", None), 
                placeholder="e.g. 24.5",
                help="Body mass index. Range: 10.0 to 100.0"
            )
            
            hyp_options = ["No", "Yes", "Unknown"]
            hyp_default = st.session_state.get("autofill_hypert", "Unknown")
            hyp_idx = hyp_options.index(hyp_default) if hyp_default in hyp_options else 2
            hypert = st.selectbox("Hypertension", options=hyp_options, index=hyp_idx, help="Hypertension present at visit")
        with med_col2:
            smokyrs = st.number_input(
                "Smoking History Years", min_value=0, max_value=87, 
                value=st.session_state.get("autofill_smokyrs", None), 
                placeholder="e.g. 0 if non-smoker",
                help="Total years smoked cigarettes. Range: 0 to 87"
            )
            dia_options = ["No", "Yes, Type I", "Yes, Type II", "Yes, other type", "Unknown"]
            dia_default = st.session_state.get("autofill_diabet", "Unknown")
            dia_idx = dia_options.index(dia_default) if dia_default in dia_options else 4
            diabet = st.selectbox("Diabetes", options=dia_options, index=dia_idx, help="Diabetes present at visit")
        with med_col3:
            alc_options = ["None/Less than once a month", "About once a MONTH", "About once a WEEK", "A few times a week", "Daily or almost daily", "Unknown"]
            alc_default = st.session_state.get("autofill_alcfreq", "Unknown")
            alc_idx = alc_options.index(alc_default) if alc_default in alc_options else 5
            alcfreq = st.selectbox("Alcohol Consumption Frequency", options=alc_options, index=alc_idx, help="During the past three months, how often did the subject have at least one drink of any alcoholic beverage such as wine, beer, malt liquor, or spirits?")

        st.markdown("<br>", unsafe_allow_html=True)

        #Cognitive tests
        st.markdown("""
            <div class="dash-card" style="margin-bottom: 5px;">
                <div class="dash-card-header">3. Cognitive & Neuropsychological Battery</div>
            </div>
        """, unsafe_allow_html=True)
        
        cog_col1, cog_col2, cog_col3 = st.columns(3)
        with cog_col1:
            naccmoca = st.number_input("MoCA Total Score :red[*]", min_value=0, max_value=30, placeholder="Enter score 0-30 (Cannot be empty)", help="Montreal Cognitive Assessment test used to spot early signs of cognitive decline. It checks things like attention, memory, and language. Range: 0 to 30")
            cdrsum = st.number_input("CDR Sum of Boxes", min_value=0.0, max_value=18.0, value=None, placeholder="e.g. 0.5", step=0.5, help="Clinical Dementia Rating to track the severity of dementia. Range: 0.0 to 18.0")
        with cog_col2: 
            minttots = st.number_input("MINT Total Score", min_value=0, max_value=32,value=None, placeholder="e.g. 28", help="Multilingual Naming Test that measures language ability. Range: 0 to 32")
            naccgds = st.number_input("Geriatric Depression Scale (GDS)", min_value=0, max_value=15,value=None, placeholder="e.g. 3", help="Geriatric Depression Scale used to screen for depression in older adults. Range: 0 to 15")
        with cog_col3:
            traila = st.number_input("Trail Making Test A (Sec)", min_value=0, max_value=150,value=None, placeholder="e.g. 45", help="Total number of seconds to complete Trail Making Test Part A that measures visual scanning and basic motor speed. Range: 0 to 150")
            trailb = st.number_input("Trail Making Test B: (Sec)", min_value=0, max_value=300,value=None, placeholder="e.g. 120", help="Total number of seconds to complete Trail Making Test Part B that measures task-switching flexibility. Range: 0 to 300")

        st.markdown("<br>", unsafe_allow_html=True)

        
        #FAQ section
        st.markdown("""
            <div class="dash-card" style="margin-bottom: 5px;">
                <div class="dash-card-header">4. Functional Activities Questionnaire (FAQ Assessment)</div>
            </div>
        """, unsafe_allow_html=True)
                
        faq_col1, faq_col2, faq_col3 = st.columns(3)
        faq_options = ["Normal", "Has Difficulty", "Requires Assistance", "Dependent", "Unknown"]
        with faq_col1:
            bills = st.selectbox("Managing Financial Bills", options=faq_options, 
                                 help="In the past four weeks, did the subject have any difficulty or need help with: Writing checks, paying bills, or balancing a checkbook")
            taxes = st.selectbox("Assembling Tax Records", options=faq_options, 
                                 help="In the past four weeks, did the subject have any difficulty or need help with: Assembling tax records, business affairs, or other papers")
            shopping = st.selectbox("Shopping Alone", options=faq_options,
                                    help="In the past four weeks, did the subject have any difficulty or need help with: Shopping alone for clothes, household necessities, or groceries")
        with faq_col2:
            games = st.selectbox("Playing Games of Skill/Hobbies", options=faq_options, 
                                 help="In the past four weeks, did the subject have any difficulty or need help with: Playing a game of skill such as bridge or chess, working on a hobby")
            mealprep = st.selectbox("Preparing Balanced Meals", options=faq_options, 
                                    help="In the past four weeks, did the subject have any difficulty or need help with: Preparing a balanced meal")
            events = st.selectbox("Tracking Current Events", options=faq_options, 
                                  help="In the past four weeks, did the subject have any difficulty or need help with: Keeping track of current events")
        with faq_col3:
            payattn = st.selectbox("Paying Attention / Conversing", options=faq_options, 
                                   help="In the past four weeks, did the subject have any difficulty or need help with: Paying attention to and understanding a TV program, book, or magazine")
            remdates = st.selectbox("Remembering Dates/Appointments", options=faq_options, 
                                    help="In the past four weeks, did the subject have any difficulty or need help with:Remembering appointments, family occasions, holidays, medications")
            travel = st.selectbox("Traveling Out of Neighborhood", options=faq_options,
                                  help="In the past four weeks, did the subject have any difficulty or need help with: Traveling out of the neighborhood, driving, or arranging to take public transportation")

        st.markdown("<br>", unsafe_allow_html=True)

        #genetic section
        st.markdown("""
            <div class="dash-card" style="margin-bottom: 5px;">
                <div class="dash-card-header">5. Genetic Profile</div>
            </div>
        """, unsafe_allow_html=True)
        
        gen_col1, gen_col2 = st.columns([1, 2])
        with gen_col1:
            apoe_options = ["No", "1 copy", "2 copies", "Unknown"]
            apoe_default = st.session_state.get("autofill_ne4s", "Unknown")
            apoe_idx = apoe_options.index(apoe_default) if apoe_default in apoe_options else 3
            naccne4s = st.selectbox("APOE ε4 Allele Count", options=apoe_options, index=apoe_idx, 
                        help="Number of copies of the APOE e4 allele present.")
        with gen_col2:
            st.markdown("<p style='color:#777; padding-top:25px;'>The presence of APOE ε4 alleles represents a genetic risk factor sequence evaluation for late-onset Alzheimer's Disease.</p>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        #Submit button form
        submit_btn = st.form_submit_button("Run Prediction", type="primary", use_container_width=True)
        
    if submit_btn:
        #check cdrsum format
        if cdrsum in [16.5, 17.5]:
            st.error(
                f"❌ **Invalid Assessment Value:** A CDR Sum of Boxes score of `{cdrsum}` is mathematically impossible under standard clinical protocols. "
                "Please review the six individual domain ratings and correct the total sum before proceeding."
            )
            st.stop()
        if cdrsum is not None and (cdrsum * 2) % 1 != 0:
            st.error("❌ **Invalid Score:** CDR Sum of Boxes must strictly be in increments of 0.5 (e.g., 0.0, 0.5, 1.0, 1.5...). Please re-enter.")
            st.stop()


        missing_fields = []
        if naccbmi is None: missing_fields.append("Body Mass Index (BMI)")
        if smokyrs is None: missing_fields.append("Smoking History Years")

        if cdrsum is None: missing_fields.append("CDR Sum of Boxes")
        if minttots is None: missing_fields.append("MINT Total Score")
        if naccgds is None: missing_fields.append("Geriatric Depression Scale (GDS)")
        if traila is None: missing_fields.append("Trail Making Test A")
        if trailb is None: missing_fields.append("Trail Making Test B")

        if race == "Unknown": missing_fields.append("Race Category")
        if bills == "Unknown": missing_fields.append("FAQ: Managing Financial Bills")
        if taxes == "Unknown": missing_fields.append("FAQ: Assembling Tax Records")
        if shopping == "Unknown": missing_fields.append("FAQ: Shopping Alone")
        if games == "Unknown": missing_fields.append("FAQ: Playing Games/Hobbies")
        if mealprep == "Unknown": missing_fields.append("FAQ: Preparing Balanced Meals")
        if events == "Unknown": missing_fields.append("FAQ: Tracking Current Events")
        if payattn == "Unknown": missing_fields.append("FAQ: Paying Attention / Conversing")
        if remdates == "Unknown": missing_fields.append("FAQ: Remembering Dates/Appointments")
        if travel == "Unknown": missing_fields.append("FAQ: Traveling Out of Neighborhood")
        if naccne4s == "Unknown": missing_fields.append("APOE ε4 Allele Count")
        if hypert == "Unknown":missing_fields.append("Hypertension")
        if diabet == "Unknown":missing_fields.append("Diabetes History")
        if alcfreq == "Unknown":missing_fields.append("Alcohol Consumption Frequency")
        if maristat == "Unknown":missing_fields.append("Marital Status")

        if missing_fields:
        #if features are missing, block execution
            show_accuracy_warning_modal(missing_fields)
        else:
            st.session_state.bypass_approved = True
        
    if st.session_state.get("bypass_approved", False):
        st.session_state.bypass_approved = False

        st.success("Form submitted!")

        with st.spinner("Processing clinical features through Random Forest model..."):
            
            #map text to numerical
            faq_map = {"Normal": "0.0", "Has Difficulty": "1.0", "Requires Assistance": "2.0", "Dependent": "3.0", "Unknown": np.nan}
            sex_map = {"Male": "1", "Female": "2"}
            apoe_map = {"No": "0.0", "1 copy": "1.0", "2 copies": "2.0", "Unknown": np.nan}
            hypert_map = {"No": "0.0", "Yes": "1.0", "Unknown": np.nan}
            diabet_map = {"No": "0.0", "Yes, Type I": "1.0", "Yes, Type II": "2.0", "Yes, other type": "3.0", "Unknown": np.nan}
            race_map = {"White": "1.0", "Black/African American": "2.0", "American Indian/Alaska Native": "3.0", "Native Hawaiian/Other Pacific Islander": "4.0", "Asian": "5.0", "Other/Multiracial": "50.0", "Unknown": np.nan}
            maristat_map = {"Married": "1.0", "Widowed": "2.0", "Divorced": "3.0", "Separated": "4.0", "Never Married": "5.0", "Living as married/domestic partner": "6.0", "Unknown": np.nan}
            alcfreq_map = {"None/Less than once a month": "0.0", "About once a MONTH": "1.0", "About once a WEEK": "2.0", "A few times a week": "3.0", "Daily or almost daily": "4.0", "Unknown": np.nan}


            #convert none to nan
            clean_bmi = naccbmi if naccbmi is not None else np.nan
            clean_smokyrs = smokyrs if smokyrs is not None else np.nan
            clean_cdrsum = cdrsum if cdrsum is not None else np.nan
            clean_minttots = minttots if minttots is not None else np.nan
            clean_naccgds = naccgds if naccgds is not None else np.nan
            clean_traila = traila if traila is not None else np.nan
            clean_trailb = trailb if trailb is not None else np.nan
            
            
            #make column names match best_rf_model.pkl features
            inputs = {
                "NACCAGE": naccage, "SEX": sex_map[sex], "EDUC": educ, "NACCBMI": clean_bmi,
                "HYPERT": hypert_map[hypert], "DIABET": diabet_map[diabet], "SMOKYRS": clean_smokyrs, "ALCFREQ": alcfreq_map[alcfreq],
                "NACCMOCA": naccmoca, "CDRSUM": clean_cdrsum, 
                "MINTTOTS": clean_minttots, "NACCGDS": clean_naccgds,
                "MARISTAT": maristat_map[maristat], "RACE": race_map[race],
                "TRAILA": clean_traila, "TRAILB": clean_trailb,
                "BILLS": faq_map[bills], "TAXES": faq_map[taxes], "SHOPPING": faq_map[shopping],
                "GAMES": faq_map[games], "MEALPREP": faq_map[mealprep], "EVENTS": faq_map[events],
                "PAYATTN": faq_map[payattn], "REMDATES": faq_map[remdates], "TRAVEL": faq_map[travel],
                "NACCNE4S": apoe_map[naccne4s]
            }

            #imputation if empty
            all_training_columns = list(saved_imputer['con_median'].keys() | saved_imputer['cat_mode'].keys())                
            input_data = pd.DataFrame(np.nan, index=[0], columns=all_training_columns)
            
            for col, val in inputs.items():
                if col in input_data.columns:
                    input_data.at[0, col] = val

            for col in input_data.columns:
                if pd.isnull(input_data.at[0, col]):
                    if col in saved_imputer['con_median']:
                        input_data[col] = input_data[col].fillna(saved_imputer['con_median'][col])
                    elif col in saved_imputer['cat_mode']:
                        input_data[col] = input_data[col].fillna(saved_imputer['cat_mode'][col])

            #encode
            encode_cat = list(encoder.feature_names_in_)
            encoded_array = encoder.transform(input_data[encode_cat])
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(encode_cat))
            con_df = input_data.drop(columns=encode_cat)
            final_feature = pd.concat([con_df, encoded_df], axis=1)

            #arrange feature order
            if training_feature_order is not None:
                final_feature = final_feature[training_feature_order]
            elif hasattr(saved_scaler, "feature_names_in_"):
                final_feature = final_feature[saved_scaler.feature_names_in_]

            #scale
            scaled_features = saved_scaler.transform(final_feature)
            
            #then send to model to predict
            prediction_code = best_rf_model.predict(scaled_features)[0]
            prediction_proba = best_rf_model.predict_proba(scaled_features)[0]

            #clasiify
            if prediction_code == 0:
                status_label = "Normal Cognition"
                brand_color = "#15877B"      
                bg_color = "rgba(21, 135, 123, 0.05)"
                status_icon = "bi-check-circle"
                primary_bar_idx = 0          #index for Normal bar
            elif prediction_code == 1:
                status_label = "Mild Cognitive Impairment (MCI)"
                brand_color = "#E67E22"     
                bg_color = "rgba(230, 126, 34, 0.05)"
                status_icon = "bi-exclamation-circle"
                primary_bar_idx = 1          #for MCI bar
            else:  #ad
                status_label = "Alzheimer's Disease (AD) Dementia"
                brand_color = "#FF7272"      
                bg_color = "rgba(231, 76, 60, 0.05)"
                status_icon = "bi-exclamation-triangle"
                primary_bar_idx = 2         #for ad bar
            
            #calculate prob for prob table
            prob_normal = prediction_proba[0] * 100
            prob_mci = prediction_proba[1] * 100 
            prob_ad = prediction_proba[2] * 100
            
            #result section shown-------------------------------------------------
            
            st.markdown("""
                <h2 class="main-header" style='display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
                    <i class="bi bi-bar-chart-line" style="color: #2D323E;"></i> Cognitive Assessment Outcomes
                </h2>
            """, unsafe_allow_html=True)
            st.markdown("The system predicts the cognitive condition based on the subject's assessment information and provides the key factors influencing the prediction result.")
            
            #columns for the diagnosis and probability part
            out_col1, out_col2 = st.columns([4, 6], gap="large")
            
            with out_col1:
                #classification result status
                st.markdown(f"""
                    <div style="background-color: #fff; border: 3px solid {brand_color}; border-radius: 6px; padding: 20px; text-align: center;">
                        <i class="bi {status_icon}" style="font-size: 28px; color: {brand_color}; display: block; margin-bottom: 5px;"></i>
                        <div style="font-size: 13px; color: #000; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
                            Predicted Cognitive Status:
                        </div>
                        <div style="font-size: 20px; font-weight: 700; color: {brand_color}; margin-top: 5px; margin-bottom: 3px;">
                            {status_label}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            with out_col2:
                #prob table
                st.markdown("<p style='font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #2D323E;'>Model Class Probabilities:</p>", unsafe_allow_html=True)
                
                chart_df = pd.DataFrame({
                    "Diagnostic Category": ["Normal Cognition", "Mild Cognitive Impairment (MCI)", "Alzheimer's Disease (AD) Dementia"],
                    "Numeric_Value": [prob_normal, prob_mci, prob_ad],
                    "Confidence Weight": [f"{prob_normal:.2f}%", f"{prob_mci:.2f}%", f"{prob_ad:.2f}%"]
                })
                #only the status bar show colour other bar gray
                bar_colors = ["#D2D2D2"] * len(chart_df)
                bar_colors[primary_bar_idx] = brand_color

                short_labels = ["Normal", "MCI", "AD Dementia"]

                fig_probs = px.bar(
                    chart_df, 
                    x=short_labels, 
                    y="Numeric_Value",
                    text=chart_df["Confidence Weight"]
                )
                
                fig_probs.update_traces(
                    marker_color=bar_colors,              
                    textposition="outside",               
                    textfont=dict(size=12, color="#2D323E", weight="bold"), # Text dark adjustment
                    cliponaxis=False                      
                )
                
                fig_probs.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        title="", 
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="Confidence Level (%)", 
                        tickfont=dict(color="#525151"),
                        range=[0, 110], 
                        showgrid=True, 
                        gridcolor="rgba(0,0,0,0.05)" 
                    ),
                    margin=dict(l=10, r=10, t=25, b=10),
                    height=185,
                    showlegend=False
                )
                #to show it
                st.plotly_chart(fig_probs, use_container_width=True, config={"displayModeBar": False})
            
            #SHAP SECTION ----------------
            st.markdown("""
                <h4 style='display: flex; align-items: center; gap: 10px; '>
                    <i class="bi bi-cpu" style="color: #2D323E;"></i> SHAP Interpretation:
                </h4>
                <p style='font-size: 16px; color: #2D323E; margin-bottom: 10px;'>
                    SHAP analysis shows how <span style="font-weight:600;">top 10 features</span> contribute to the prediction outcome.
                </p>
            """, unsafe_allow_html=True)

            #use shap interpret
            #Convert scaled array into a DataFrame
            scaled_features_df = pd.DataFrame(
                scaled_features, 
                columns=final_feature.columns
            )

            shap_explain = shap.TreeExplainer(best_rf_model)
            raw_shap_values = shap_explain.shap_values(scaled_features_df)
            current_shap = raw_shap_values[:, :, prediction_code][0]
            
            shap_data = pd.DataFrame({
                "Risk Factor": final_feature.columns,
                "Signed_Impact": current_shap
            })


            #since the variable encoded, to understand, let map to names
            readable_names = {
                # Sex
                "SEX_1": "Biological Sex: Male",
                "SEX_2": "Biological Sex: Female",
                
                #Hypertension & Diabetes
                "HYPERT_1.0": "Presence of Hypertension: Yes",
                "HYPERT_0.0": "Presence of Hypertension: No",
                "DIABET_0.0": "History of Diabetes: No",
                "DIABET_1.0": "Diabetes: Type I",
                "DIABET_2.0": "Diabetes: Type II",
                "DIABET_3.0": "Diabetes: Other Type",

                #Race 
                "RACE_1.0": "Race: White",
                "RACE_2.0": "Race: Black/African American",
                "RACE_3.0": "Race: American Indian/Alaska Native",
                "RACE_4.0": "Race: Native Hawaiian/Pacific Islander",
                "RACE_5.0": "Race: Asian",
                "RACE_50.0": "Race: Other/Multiracial",
                
                #Marital Status 
                "MARISTAT_1.0": "Marital Status: Married",
                "MARISTAT_2.0": "Marital Status: Widowed",
                "MARISTAT_3.0": "Marital Status: Divorced",
                "MARISTAT_4.0": "Marital Status: Separated",
                "MARISTAT_5.0": "Marital Status: Never Married",
                "MARISTAT_6.0": "Marital Status: Domestic Partner",
                
                #Alcohol Frequency Mappings
                "ALCFREQ_0.0": "Alcohol: None / < Once a Month",
                "ALCFREQ_1.0": "Alcohol: ~ Once a Month",
                "ALCFREQ_2.0": "Alcohol: ~ Once a Week",
                "ALCFREQ_3.0": "Alcohol: A Few Times a Week",
                "ALCFREQ_4.0": "Alcohol: Daily / Almost Daily",
                
                #faq
                "BILLS_0.0": "FAQ (Bills): Normal Performance",
                "BILLS_1.0": "FAQ (Bills): Has Difficulty",
                "BILLS_2.0": "FAQ (Bills): Requires Assistance",
                "BILLS_3.0": "FAQ (Bills): Dependent",
                
                "TAXES_0.0": "FAQ (Taxes): Normal Performance",
                "TAXES_1.0": "FAQ (Taxes): Has Difficulty",
                "TAXES_2.0": "FAQ (Taxes): Requires Assistance",
                "TAXES_3.0": "FAQ (Taxes): Dependent",
                
                "SHOPPING_0.0": "FAQ (Shopping): Normal Performance",
                "SHOPPING_1.0": "FAQ (Shopping): Has Difficulty",
                "SHOPPING_2.0": "FAQ (Shopping): Requires Assistance",
                "SHOPPING_3.0": "FAQ (Shopping): Dependent",

                "REMDATES_0.0": "FAQ (Appointments): Normal Performance",
                "REMDATES_1.0": "FAQ (Appointments): Has Difficulty",
                "REMDATES_2.0": "FAQ (Appointments): Requires Assistance",
                "REMDATES_3.0": "FAQ (Appointments): Dependent",
                
                "TRAVEL_0.0": "FAQ (Travel): Normal Performance",
                "TRAVEL_1.0": "FAQ (Travel): Has Difficulty",
                "TRAVEL_2.0": "FAQ (Travel): Requires Assistance",
                "TRAVEL_3.0": "FAQ (Travel): Dependent",

                "GAMES_0.0": "FAQ (Games): Normal Performance",
                "GAMES_1.0": "FAQ (Games): Has Difficulty",
                "GAMES_2.0": "FAQ (Games): Requires Assistance",
                "GAMES_3.0": "FAQ (Games): Dependent",

                "MEALPREP_0.0": "FAQ (Meal Preparation): Normal Performance",
                "MEALPREP_1.0": "FAQ (Meal Preparation): Has Difficulty",
                "MEALPREP_2.0": "FAQ (Meal Preparation): Requires Assistance",
                "MEALPREP_3.0": "FAQ (Meal Preparation): Dependent",
                
                "EVENTS_0.0": "FAQ (Event Tracking): Normal Performance",
                "EVENTS_1.0": "FAQ (Event Tracking): Has Difficulty",
                "EVENTS_2.0": "FAQ (Event Tracking): Requires Assistance",
                "EVENTS_3.0": "FAQ (Event Tracking): Dependent",

                "PAYATTN_0.0": "FAQ (Paying Attention): Normal Performance",
                "PAYATTN_1.0": "FAQ (Paying Attention): Has Difficulty",
                "PAYATTN_2.0": "FAQ (Paying Attention): Requires Assistance",
                "PAYATTN_3.0": "FAQ (Paying Attention): Dependent",

                "NACCNE4S_0.0": "APOE ε4 Allele no copy",
                "NACCNE4S_1.0": "APOE ε4 Allele 1 copy",
                "NACCNE4S_2.0": "APOE ε4 Allele 2 copies",

                #other numerical features
                "NACCAGE": "Age",
                "EDUC": "Education History (Years)",
                "SMOKYRS": "Smoking History (Years)",
                "NACCBMI": "Body Mass Index (BMI)",
                "NACCMOCA": "MoCA Total Score",
                "CDRSUM": "CDR Sum of Boxes",
                "MINTTOTS": "MINT Total Score",
                "NACCGDS": "Geriatric Depression Scale (GDS)",
                "TRAILA": "Trail Making Test A (Sec)",
                "TRAILB": "Trail Making Test B (Sec)"
            }
            shap_data["Risk Factor"] = shap_data["Risk Factor"].map(readable_names).fillna(shap_data["Risk Factor"])

            #sort data so the top 10 influential factor appears at the top bar chart
            shap_data["Absolute_Impact"] = shap_data["Signed_Impact"].abs()
            shap_data = shap_data.sort_values(by="Absolute_Impact", ascending=False).head(10)
            shap_data = shap_data.sort_values(by="Absolute_Impact", ascending=True)
            #if negative impact go left, positive go right
            if prediction_code == 0:  #Normal Cognition
                shap_data["Direction"] = shap_data["Signed_Impact"].apply(
                    lambda x: "Reduces Normal Class Likelihood" if x < 0 else "Increases Normal Class Likelihood (Healthy)"
                )
            elif prediction_code == 1: #MCI
                shap_data["Direction"] = shap_data["Signed_Impact"].apply(
                    lambda x: "Pushes Away from MCI" if x < 0 else "Contributes to MCI Risk"
                )
            else: #AD Dementia
                shap_data["Direction"] = shap_data["Signed_Impact"].apply(
                    lambda x: "Points to Healthy Cognition" if x < 0 else "Points to Dementia Risk"
                )

            #build a horizontal bar chart using Plotly
            ui_color_map = {
                # Class 2 (Dementia) Strings
                "Points to Dementia Risk": "#FF7272",         
                "Points to Healthy Cognition": "#407CD1",     
                
                # Class 1 (MCI) Strings
                "Contributes to MCI Risk": "#FF7272",         
                "Pushes Away from MCI": "#407CD1",            
                
                # Class 0 (Normal) Strings
                "Increases Normal Class Likelihood (Healthy)": "#407CD1", 
                "Reduces Normal Class Likelihood": "#FF7272"  
            }
            fig_shap = px.bar(
                shap_data, 
                x="Signed_Impact", 
                y="Risk Factor", 
                orientation="h",
                color="Direction",
                color_discrete_map=ui_color_map,
                text=shap_data["Signed_Impact"].apply(lambda x: f"{x:+.2f}")
            )

            fig_shap.update_traces(
                textposition="outside",
                textfont=dict(size=14, color="#2D323E", weight="bold"),
                cliponaxis=False
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title=dict(
                        text="SHAP Value",
                        font=dict(color="#525151", size=12)
                    ),
                    showgrid=True, 
                    gridcolor="rgba(0,0,0,0.05)", 
                    zeroline=True,
                    zerolinecolor="#2D323E",
                    zerolinewidth=1.5,
                    tickfont=dict(color="#2D323E"),
                    range=[-0.35, 0.35],
                    tickformat=".2f" 
                ),
                yaxis=dict(
                    title="",
                    showgrid=False, 
                    tickfont=dict(color="#2D323E", size=14, weight="normal")
                ),
                font=dict(color="#2D323E"), 
                legend=dict(title_text="Impact Direction", orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=20, b=40),
                height=340
            )

            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

            #clinical insight below the chart
            class_status = int(prediction_code)
            current_confidence = prediction_proba[class_status] * 100

            # sentences
            if prediction_code == 0:    # Normal 
                insight_text = (
                    f"The subject shows good cognitive health. "
                    f"showing a <strong style='font-weight:700;color: #15877B;'>{current_confidence:.2f}%, probability of Normal Cognition</strong>, "
                    f"closely aligning them with healthy individuals."
                    f"Routine follow-up assessment is recommended to monitor baseline stability."
                )
            elif prediction_code == 1:  # Mild Cognitive Impairment (MCI)
                insight_text = (
                    f"The model detects early-stage cognitive decline. The system calculates a "
                    f"<strong style='font-weight:700;color: #E67E22;'>{current_confidence:.2f}% probability of Mild Cognitive Impairment (MCI).</strong> "
                    f"This signal points toward early decline, typically indicating a need for localised cognitive exercises, "
                    f"lifestyle modifications and a diagnostic review within 6 months."
                )
            else:      #AD Dementia
                insight_text = (
                    f"The classification shows clear signs of cognitive decline "
                    f"within <strong style='font-weight:700;color: #FF7272;'>{current_confidence:.2f}% probability of Alzheimer's Disease (AD) Dementia.</strong> "
                    f"The combined effect of neuropsychological core deficits and daily functional dependence supports this classification."
                    f"Immediate therapeutic intervention and supportive clinical planning are recommended."
                )
            #show
            st.markdown(f"""
                <div style="background-color: rgba(45, 50, 62, 0.03); border: 2px dashed #2D323E; border-radius: 6px; padding: 12px; margin-top: 10px;">
                    <p style="font-size: 16px; color: #2D323E; margin: 0; font-weight: 600;">
                        <i class="bi bi-info-circle-fill" style="margin-right: 5px;"></i> 
                        <span style="font-weight:700;"> Clinical Insight: </span> </br>  {insight_text}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</br>", unsafe_allow_html=True)

            time_str = now_my.strftime("%I:%M %p")

            #convert shap to png to put on report
            try:
                #adjust size
                fig_shap.update_layout(
                    margin=dict(l=200, r=20, t=40, b=40), 
                    width=600,                            
                    height=350,                           
                    autosize=False,
                    yaxis=dict(
                        title="",
                        showgrid=False, 
                        tickfont=dict(color="#2D323E", size=12, weight="normal")
                    )
                )
                fig_shap.update_traces(
                    textposition="outside",
                    textfont=dict(size=11, color="#2D323E", weight="bold"),
                    cliponaxis=False
                )
                
                img_bytes = fig_shap.to_image(format="png", width=600, height=350, scale=2)
                
                #encode image bytes to a text string
                encoded_shap = base64.b64encode(img_bytes).decode("utf-8")
                
                #create image
                shap_img_html = f'<img src="data:image/png;base64,{encoded_shap}" style="width: 600px; height: auto; margin-top: 10px; border: 1px solid #e2e8f0;" />'
            except Exception as image_err:
                #if fails
                shap_img_html = f'<p style="color: red; font-style: italic;">Visual compilation skipped: {image_err}</p>'

            
            disp_educ = f"{int(educ)}" if educ is not None and not pd.isna(educ) else "Not Provided"
            disp_bmi = f"{float(naccbmi):.1f}" if naccbmi is not None and not pd.isna(naccbmi) else "Not Provided"
            disp_smokyrs = f"{int(smokyrs)}" if smokyrs is not None and not pd.isna(smokyrs) else "Not Provided"
            
            disp_cdrsum = f"{float(cdrsum):.1f}" if cdrsum is not None and not pd.isna(cdrsum) else "Not Provided"
            disp_mint = f"{int(minttots)}" if minttots is not None and not pd.isna(minttots) else "Not Provided"
            disp_gds = f"{int(naccgds)}" if naccgds is not None and not pd.isna(naccgds) else "Not Provided"
            disp_traila = f"{int(traila)}" if traila is not None and not pd.isna(traila) else "Not Provided"
            disp_trailb = f"{int(trailb)}" if trailb is not None and not pd.isna(trailb) else "Not Provided"
            
            #set timestamp and reference id to save in database
            combined_timestamp = f"{malaysia_str}, {time_str}"
            del st.session_state.current_ref_id
            unique_suffix = str(uuid.uuid4()).split("-")[0].upper()
            current_year = datetime.now().year
            st.session_state.current_ref_id = f"REF-{current_year}-{unique_suffix}"
            
            
            #clinical report design
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
                    
                    @page {{
                        size: a4;
                        margin: 20mm 15mm 20mm 15mm;
                    }}
                    
                    body {{
                        font-family: 'Roboto', sans-serif;
                        color: #2D323E;
                        line-height: 1.4;
                        font-size: 10pt;
                    }}

                    .header-table {{
                        width: 100%;
                        margin-bottom: 3px;
                        border-bottom: 1px solid #2D323E;
                    }}
                    
                    .title {{
                        font-size: 18pt;
                        font-weight: 700;
                        color: #15877B;
                        padding-left: 5px;
                        width: 500px;
                    }}
                    
                    .timestamp {{
                        text-align: right;
                        margin-right:0px;
                        margin-bottom:0px;
                        font-size: 10pt;
                        padding: 5px;
                        color: #717171;
                    }}
                    
                    .status-box {{
                        border: 2px solid {brand_color};
                        text-align: center;
                        background-color: #ffffff;
                        margin-left:60px;
                        margin-right:60px;
                        padding-top:15px;
                    }}
                    
                    .status-title {{
                        font-size: 12pt;
                        font-weight: 700;
                        color: #2D323E;
                    }}
                    
                    .status-value {{
                        font-size: 14pt;
                        font-weight: 700;
                        color: {brand_color};
                    }}
                    
                    .section-heading {{
                        font-size: 12pt;
                        font-weight: 700;
                        text-transform: uppercase;
                        color: #fff;
                        background-color: #2D323E;
                        padding-left:10px;
                        padding-top:10px;
                        margin:5px;
                    }}
                    
                    .grid-table {{
                        width: 100%;
                        margin-bottom: 10px;
                    }}
                    
                    .grid-table td {{
                        width: 50%;
                        padding: 4px 5px;
                        font-size: 11pt;
                    }}
                    
                    .data-label {{
                        font-weight: 400;
                        color: #4a5568;
                    }}
                    
                    .data-value {{
                        font-weight: 600;
                        color: #2D323E;
                    }}
                    
                    .insight-card {{
                        padding: 12px; 
                        margin: 10px;
                        font-size: 12pt;
                        margin-top:25px;
                        text-align:justify;
                    }}
                    
                    .insight-heading {{
                        font-weight: 700;
                        text-transform: uppercase;
                        color: #2D323E;
                        margin-bottom: 5px;
                    }}
                    
                    .footer-notice {{
                        margin-top: 35px;
                        font-size: 9pt;
                        color: #15877B;
                        text-align: center;
                        border-top: 0.5px solid #2D323E;
                        padding-top: 10px;
                    }}
                </style>
            </head>
            <body>

            <table class="header-table">
                <tr>
                    <td class="title">COGNEUTEST </br> <span style="font-size: 16pt; font-weight: 500; margin-top: 4px;">Cognitive Decline Assessment System</span></td>
                    <td class="timestamp">
                        {malaysia_str} </br> {time_str} MYT
                    </td>
                </tr>
            </table>
            <div>
                <p style="font-size: 14px; font-weight: bold; padding:7px; color: #2D323E;">CLINICAL REFERENCE ID:
                <span style="color: #15877B;">{st.session_state.current_ref_id}</span></br>
                <span style="font-size: 14px; font-weight: bold; padding:7px; color: #2D323E;">SUBJECT ID:
                <span style="color: #15877B;">{patient_master_id}</span></p>
            </div>
            <div class="status-box">
                <p class="status-title">Predicted Cognitive Status:</br></br>
                <span class="status-value" style="padding-top:10px;">{status_label}</span></p>
            </div>

            <div class="section-heading">Model Classification Confidence</div>
            <table class="grid-table">
                <tr>
                    <td><span class="data-label">Normal Cognition Probability:</span> <span class="data-value">{prediction_proba[0]*100:.2f}%</span></td>
                    <td><span class="data-label">Mild Cognitive Impairment (MCI):</span> <span class="data-value">{prediction_proba[1]*100:.2f}%</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Alzheimer's Disease (AD) Probability:</span> <span class="data-value">{prediction_proba[2]*100:.2f}%</span></td>
                    <td></td>
                </tr>
            </table>

            <div class="section-heading">Patient Demographics & Medical Condition</div>
            <table class="grid-table">
                <tr>
                    <td><span class="data-label">Age:</span> <span class="data-value">{naccage} years old</span></td>
                    <td><span class="data-label">Biological Sex:</span> <span class="data-value">{sex}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Education Years:</span> <span class="data-value">{disp_educ} years</span></td>
                    <td><span class="data-label">Race Classification:</span> <span class="data-value">{race}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Marital Status:</span> <span class="data-value">{maristat}</span></td>
                    <td></td>
                </tr>
            </table>

            <div class="section-heading">Lifestyle, Medical & Genetic Factor</div>
            <table class="grid-table">
                <tr>
                    <td><span class="data-label">Calculated BMI Index:</span> <span class="data-value">{disp_bmi}</span></td>
                    <td><span class="data-label">APOE e4 allele count:</span> <span class="data-value">{naccne4s}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Hypertension History:</span> <span class="data-value">{hypert}</span></td>
                    <td><span class="data-label">Diabetes History:</span> <span class="data-value">{diabet}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Smoking History:</span> <span class="data-value">{disp_smokyrs} years</span></td>
                    <td><span class="data-label">Alcohol Frequency:</span> <span class="data-value">{alcfreq}</span></td>
                </tr>
            </table>

            <div class="section-heading">Neuropsychological Cognitive Metrics</div>
            <table class="grid-table">
                <tr>
                    <td><span class="data-label">MoCA Total Score:</span> <span class="data-value">{naccmoca} / 30</span></td>
                    <td><span class="data-label">CDR Sum of Boxes (CDRSUM):</span> <span class="data-value">{disp_cdrsum}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Multilingual Naming (MINT):</span> <span class="data-value">{disp_mint}</span></td>
                    <td><span class="data-label">Geriatric Depression (GDS):</span> <span class="data-value">{disp_gds}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Trail Making Test Part A:</span> <span class="data-value">{disp_traila} sec</span></td>
                    <td><span class="data-label">Trail Making Test Part B:</span> <span class="data-value">{disp_trailb} sec</span></td>
                </tr>
            </table>

            <div class="section-heading">Functional Activities Questionnaire (FAQ)</div>
            <table class="grid-table">
                <tr>
                    <td><span class="data-label">Writing Checks/Bills:</span> <span class="data-value">{bills}</span></td>
                    <td><span class="data-label">Assembling Tax Records:</span> <span class="data-value">{taxes}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Independent Shopping:</span> <span class="data-value">{shopping}</span></td>
                    <td><span class="data-label">Engaging in Hobbies/Games:</span> <span class="data-value">{games}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Meal Preparation:</span> <span class="data-value">{mealprep}</span></td>
                    <td><span class="data-label">Tracking Current Events:</span> <span class="data-value">{events}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Attention/Comprehension:</span> <span class="data-value">{payattn}</span></td>
                    <td><span class="data-label">Remembering Dates/Appointments:</span> <span class="data-value">{remdates}</span></td>
                </tr>
                <tr>
                    <td><span class="data-label">Community Travel Independence:</span> <span class="data-value">{travel}</span></td>
                    <td></td>
                </tr>
            </table>
            <hr style="border: 0.5px solid #2D323E;">
            <div class="section-heading">SHAP Feature Contribution Interpretation Chart</div>
            <div style="text-align: center; margin-top: 10px; margin-bottom: 15px; width: 100%;">
                <p style="font-size: 10pt; padding-left:10px; color: #4a5568; margin-bottom: 8px; text-align: left;">
                    The chart below displays feature attributions calculated using SHAP values and shows how <strong>top 10 clinical features</strong> drove the model's diagnostic choice for this patient.
                </p>
                
                {shap_img_html}
            </div>
            <hr style="border: 0.5px solid #2D323E;">
            <div class="insight-card">
                <p><span style="font-weight:700;"> Clinical Insight: </span> </br>{insight_text}</p>
            </div>

            <div class="footer-notice">
                <strong>CONFIDENTIAL MEDICAL DOCUMENTATION</strong><br>
                This automated summary record was exported via the COGNEUTEST processing engine. 
                Risk analysis values are predicted using a trained Random Forest model.
            </div>

            </body>
            </html>
            """
            #convert html to pdf
            expander_title_html ="""
            <span style="color: #15877B; font-weight: 600; display: inline-flex; align-items: center; gap: 8px;">
                <i class="bi bi-download"></i> Export Patient Record
            </span>
            """
            st.info("⚠️ **Notice:** Downloading this report will clear the result and cause current visualisations to reset.")
            pdf_file = convert_html_to_pdf(html_report)
            st.download_button(
                label=":material/download: Download Clinical Report (PDF)",
                data=pdf_file,
                file_name=f"COGNEUTEST_Report_{st.session_state.current_ref_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            #save record to database-------------
            if db_connection is not None:
                try:
                    sql_insert_query = """
                        INSERT INTO patient_records (
                            REFERENCE_ID, NACCID, CDRSUM, NACCMOCA, TRAILB, REMDATES, TRAILA, TAXES, 
                            TRAVEL, NACCBMI, BILLS, NACCAGE, PAYATTN, SHOPPING, ALCFREQ, EVENTS, 
                            NACCGDS, EDUC, MEALPREP, SMOKYRS, MARISTAT, GAMES, RACE, 
                            NACCNE4S, SEX, HYPERT, DIABET, MINTTOTS, STATUS, REPORT_PDF, TIMESTAMP
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?
                        )
                    """

                    record_tuple = (
                        st.session_state.current_ref_id,   # REFERENCE_ID (Primary Key)
                        patient_master_id,                 #subjectid
                        float(inputs["CDRSUM"]) if not pd.isnull(inputs["CDRSUM"]) else None,
                        float(inputs["NACCMOCA"]),
                        float(inputs["TRAILB"]) if not pd.isnull(inputs["TRAILB"]) else None,
                        float(inputs["REMDATES"]) if not pd.isna(inputs["REMDATES"]) else None,
                        float(inputs["TRAILA"]) if not pd.isnull(inputs["TRAILA"]) else None,
                        float(inputs["TAXES"]) if not pd.isna(inputs["TAXES"]) else None,
                        float(inputs["TRAVEL"]) if not pd.isna(inputs["TRAVEL"]) else None,
                        float(inputs["NACCBMI"]) if not pd.isnull(inputs["NACCBMI"]) else None,
                        float(inputs["BILLS"]) if not pd.isna(inputs["BILLS"]) else None,
                        int(float(inputs["NACCAGE"])),
                        float(inputs["PAYATTN"]) if not pd.isna(inputs["PAYATTN"]) else None,
                        float(inputs["SHOPPING"]) if not pd.isna(inputs["SHOPPING"]) else None,
                        float(inputs["ALCFREQ"]) if not pd.isna(inputs["ALCFREQ"]) else None,
                        float(inputs["EVENTS"]) if not pd.isna(inputs["EVENTS"]) else None,
                        float(inputs["NACCGDS"]) if not pd.isnull(inputs["NACCGDS"]) else None,
                        int(float(inputs["EDUC"])) if not pd.isna(inputs["EDUC"]) else None,
                        float(inputs["MEALPREP"]) if not pd.isna(inputs["MEALPREP"]) else None,
                        float(inputs["SMOKYRS"]) if not pd.isnull(inputs["SMOKYRS"]) else None,
                        float(inputs["MARISTAT"]) if not pd.isna(inputs["MARISTAT"]) else None,
                        float(inputs["GAMES"]) if not pd.isna(inputs["GAMES"]) else None,
                        float(inputs["RACE"]) if not pd.isna(inputs["RACE"]) else None,
                        float(inputs["NACCNE4S"]) if not pd.isna(inputs["NACCNE4S"]) else None,
                        int(float(inputs["SEX"])),
                        float(inputs["HYPERT"]) if not pd.isna(inputs["HYPERT"]) else None,
                        float(inputs["DIABET"]) if not pd.isna(inputs["DIABET"]) else None,
                        int(float(inputs["MINTTOTS"])) if not pd.isnull(inputs["MINTTOTS"]) else None,
                        int(float(prediction_code)),               
                        sqlite3.Binary(pdf_file),
                        combined_timestamp
                    )
                    
                    db_cursor.execute(sql_insert_query, record_tuple)
                    db_connection.commit()
                    st.success("Record successfully saved to local database!")

                    if "current_ref_id" in st.session_state:
                        del st.session_state.current_ref_id

                    if "new_patient_token" in st.session_state:
                        del st.session_state.new_patient_token

                except Exception as db_err:
                    st.error(f"❌ Database commitment execution halt: {db_err}")
     
#----------------------------dashboard page-----------------------------------------

elif st.session_state.current_page == "Dashboard":
    st.markdown(f'<h2 class="main-header">{st.session_state.current_page}</h1>', unsafe_allow_html=True)
    
    if db_connection is not None:
        try:
            df_live = pd.read_sql_query("SELECT NACCID, REFERENCE_ID, NACCAGE, SEX, STATUS, TIMESTAMP FROM patient_records", db_connection)
        except Exception:
            df_live = pd.DataFrame(columns=["NACCID", "REFERENCE_ID", "NACCAGE", "SEX", "STATUS", "TIMESTAMP"])
    else:
        df_live = pd.DataFrame(columns=["NACCID", "REFERENCE_ID", "NACCAGE", "SEX", "STATUS", "TIMESTAMP"])
    
    # Calculate total records, total patients, class count
    total_records = len(df_live)
    unique_patients = df_live["NACCID"].nunique() if total_records > 0 else 0
    mci_count = len(df_live[df_live["STATUS"] == 1])
    dementia_count = len(df_live[df_live["STATUS"] == 2])
    at_risk_total = mci_count + dementia_count

    #calculate average of risk
    at_risk_pct = (at_risk_total / total_records * 100) if total_records > 0 else 0.0
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Card 1
    with c1:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-header"><i class="bi bi-folder2-open" style="margin-right: 6px; color: #fff;"></i>Total Records</div>
            <div class="dash-card-body">
                <div>
                    <div class="dash-value">{total_records}</div>
                    <div class="dash-subtext">TOTAL VISITS RECORDED</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Card 2
    with c2:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-header"><i class="bi bi-people-fill" style="margin-right: 6px; color: #fff;"></i>Total Patients</div>
            <div class="dash-card-body">
                <div>
                    <div class="dash-value">{unique_patients}</div>
                    <div class="dash-subtext">INDIVIDUAL PATIENTS</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Card 3
    with c3:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-header"><i class="bi bi-exclamation-triangle-fill" style="margin-right: 6px; color: #fff;"></i>Total At-Risk Cases</div>
            <div class="dash-card-body">
                <div>
                    <div class="dash-value">{at_risk_total}</div>
                    <div class="dash-subtext">MCI + AD DEMENTIA CASES</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Card 4
    with c4:
        st.markdown(f"""
        <div class="dash-card">
            <div class="dash-card-header"><i class="bi bi-pie-chart-fill" style="margin-right: 6px; color: #fff;"></i>At-Risk Proportion</div>
            <div class="dash-card-body">
                <div>
                    <div class="dash-value">{at_risk_pct:.1f}%</div>
                    <div class="dash-subtext">OF CLINICAL DATABASE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        f'##### <i class="bi bi-graph-up-arrow" style="-webkit-text-stroke: 1px;margin-right: 8px; color: #15877B;"></i> <span style=color:#15877B;>Distribution Analysis</span>', 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)

    with col1:
        if total_records > 0:
            status_counts = df_live["STATUS"].value_counts().to_dict()
            df_dist = pd.DataFrame([
                {"Cognitive Status": "Normal", "Count": status_counts.get(0, 0)},
                {"Cognitive Status": "MCI", "Count": status_counts.get(1, 0)},
                {"Cognitive Status": "Dementia Risk", "Count": status_counts.get(2, 0)}
            ])
        else:
            df_dist = pd.DataFrame({"Cognitive Status": ["Normal", "MCI", "Dementia Risk"], "Count": [0, 0, 0]})

        fig_dist = px.bar(
            df_dist, x="Cognitive Status", y="Count", 
            color="Cognitive Status",
            color_discrete_map={"Normal": "#15877B", "MCI": "#FFAA00", "Dementia Risk": "#FF7272"}
        )
        fig_dist.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300, showlegend=False)

        st.markdown('<div class="dash-card"><div class="dash-card-header">Cognitive Status Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if total_records > 0:
            #use bin separate age group
            bins = [0, 60, 70, 80, 130]
            labels = ["≤60", "60-70", "71-80", "80+"]
            df_live["Age Group"] = pd.cut(df_live["NACCAGE"], bins=bins, labels=labels, include_lowest=True)
            df_demo = df_live["Age Group"].value_counts().reindex(labels).reset_index()
            df_demo.columns = ["Age Group", "Patients"]
        else:
            df_demo = pd.DataFrame({"Age Group": ["≤60", "61-70", "71-80", "80+"], "Patients": [0, 0, 0, 0]})

        fig_demo = px.pie(df_demo, names="Age Group", values="Patients", hole=0.4, color_discrete_sequence=px.colors.sequential.Teal, category_orders={"Age Group": ["≤60", "60-70", "71-80", "80+"]})
        fig_demo.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300)

        st.markdown('<div class="dash-card"><div class="dash-card-header">Patient Age Group</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_demo, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'##### <i class="bi bi-clock-history" style="-webkit-text-stroke: 1px;margin-right: 8px; color: #15877B;"></i> <span style=color:#15877B;>Recent Clinical Diagnostic Logs</span>', 
        unsafe_allow_html=True
    )

    if total_records > 0:
        display_df = df_live[["REFERENCE_ID", "NACCID", "NACCAGE", "SEX", "STATUS", "TIMESTAMP"]].iloc[::-1].copy()
        #decode to text from num
        display_df["SEX"] = display_df["SEX"].map({1: "Male", 2: "Female"})
        display_df["STATUS"] = display_df["STATUS"].map({0: "Normal Cognition", 1: "Mild Cognitive Impairment (MCI)", 2: "AD Dementia"})
        
        #rename
        display_df.columns = ["Reference ID", "Patient NACCID", "Age", "Sex", "Predicted Cognitive Status", "Recorded Date"]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ System database repository is currently empty. Complete an assessment to populate dashboard metrics.")

    st.markdown("</div>", unsafe_allow_html=True)


#----------------------------record search page-----------------------------------------
elif st.session_state.current_page == "Record Search":
    st.markdown(f'<h2 class="main-header">{st.session_state.current_page}</h1>', unsafe_allow_html=True)

    st.markdown('##### <i class="bi bi-database" style="margin-right: 5px; color: #15877B;-webkit-text-stroke: 1px;"></i> <span style="color: #15877B;">Search Clinical Records</span>', unsafe_allow_html=True)
    existing_refs = []
    existing_subids = []
    

    if db_connection is not None:
        try:
            # Fetch all unique Reference IDs
            refs_df = pd.read_sql_query("SELECT DISTINCT REFERENCE_ID FROM patient_records WHERE REFERENCE_ID IS NOT NULL AND REFERENCE_ID != ''", db_connection)
            existing_refs = refs_df["REFERENCE_ID"].tolist()
            
            # Fetch all unique Patient NACCIDs
            naccids_df = pd.read_sql_query("SELECT DISTINCT NACCID FROM patient_records WHERE NACCID IS NOT NULL AND NACCID != ''", db_connection)
            existing_naccids = naccids_df["NACCID"].tolist()
        except Exception as fetch_err:
            print(f"Search Page Fetch Error: {fetch_err}")


    search_col1, search_col2 = st.columns([1, 2])
    with search_col1:
        search_type = st.radio(
            "Lookup:",
            ["By Reference ID", "By Patient ID"],
            horizontal=False
        )
    
    with search_col2:
        if search_type == "By Reference ID":
            if existing_refs:
                search_query = st.selectbox(
                    "Select Reference ID:",
                    options=existing_refs,
                    help="Type or scroll to find a specific visit record."
                )
            else:
                st.warning("⚠️ No reference ID found in the database.")
                search_query = None
        else:
            if existing_naccids:
                search_query = st.selectbox(
                    "Select Subject ID:",
                    options=existing_naccids,
                    help="Type or scroll to see all records for a specific patient."
                )
            else:
                st.warning("⚠️ No patient IDs found in the database.")
                search_query = None

    #query serach
    if search_query:
        if db_connection is not None:
            try:
                #select lookup parameters
                if search_type == "By Reference ID":
                    sql_query = "SELECT * FROM patient_records WHERE REFERENCE_ID = ?"
                else:
                    sql_query = "SELECT * FROM patient_records WHERE NACCID = ? ORDER BY ROWID DESC"
                
                #fetch query results into pandas
                results_df = pd.read_sql_query(sql_query, db_connection, params=(search_query.strip(),))
                
                if not results_df.empty:
                    st.markdown(
                        f"""
                        <div style="background-color: #dcf1e8; color: #0F5132; border: 1px solid #BADBCC; padding: 12px 15px; border-radius: 6px; font-weight: 500; display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                            <i class="bi bi-check-circle-fill" style="font-size: 18px; color: #15877B !important;"></i>
                            <span>Found {len(results_df)} matching record row(s) inside local database.</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    #find record
                    for idx, row in results_df.iterrows():
                        ref_id = row['REFERENCE_ID']
                        patient_id = row['NACCID']
                        age = int(row['NACCAGE'])
                        sex_label = "Male" if int(row['SEX']) == 1 else "Female"
                        moca_score = row['NACCMOCA']

                        #timestamp
                        record_time = row.get('TIMESTAMP', None)
                        if pd.isnull(record_time) or str(record_time).strip() == "":
                            timestamp_display = "*Timestamp Not Recorded*"
                        else:
                            timestamp_display = f"{record_time}"
                        
                        #get status
                        status_map = {0: "Normal Cognition", 1: "Mild Cognitive Impairment (MCI)", 2: "Alzheimer's Disease Dementia"}
                        diagnostic_status = status_map.get(int(row['STATUS']), "Unknown")
                        
                        #extract the binary PDF data
                        pdf_blob_bytes = row['REPORT_PDF']

                        #take out record
                        st.markdown(f"""
                        <div class="dash-card" style="margin-top: 15px;">
                            <div class="dash-card-header" style="color: #FFFFFF; display: flex; justify-content: space-between; align-items: center;">
                                <span><i class="bi bi-file-earmark-medical" style=" -webkit-text-stroke: 1px;margin-right: 8px; color: #00D2C4 !important;"></i> RECORD: {ref_id}</span>
                                <span style="background-color: #15877B; padding: 2px 10px; border-radius: 4px; font-size: 11px;">ACTIVE RECORD</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Build layout split info cards
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.markdown(f"**Subject ID:** `{patient_id}`")
                            st.markdown(f"**Demographics:** {age} Years Old | {sex_label}")
                        with info_col2:
                            st.markdown(f"**Predicted Cognitive Status:** {diagnostic_status}")
                            st.markdown(timestamp_display)
                        with info_col3:
                            #download part
                            if pdf_blob_bytes is not None:
                                st.download_button(
                                    label=":material/download: Download Diagnostic PDF Report",
                                    data=pdf_blob_bytes,
                                    file_name=f"COGNEUTEST_Report_{ref_id}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_btn_{ref_id}_{idx}",
                                    use_container_width=True
                               )
                            else:
                                st.caption("⚠️ Report missing or empty for this record.")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("❌ No matching encounters or archived diagnostic report records found for that identifier.")
                    
            except Exception as search_err:
                st.error(f"❌ Core Search Query Aborted: {search_err}")
        else:
            st.warning("⚠️ Local operational database connector is offline. Search functions disabled.")


#----------------------------next page-----------------------------------------
elif st.session_state.current_page == "About":
    st.markdown(f'<h2 class="main-header">{st.session_state.current_page}</h1>', unsafe_allow_html=True)

    st.markdown('##### <i class="bi bi-info-circle-fill" style="margin-right: 8px; color: #15877B;"></i> <span style="color: #15877B;">Project Overview</span>', unsafe_allow_html=True)
    #intro of this project
    st.markdown("""
    <div class="dash-card" style="text-align:justify;padding: 20px; line-height: 1.6; color: #2D323E;">
        <p><strong>COGNEUTEST</strong> is an Clinical Decision Support System designed to assist healthcare professionals in the early screening and risk stratification of cognitive impairments, 
            specifically <strong>Mild Cognitive Impairment (MCI)</strong> and <strong>Dementia Risk (Alzheimer's Disease)</strong>.</p>
        <p>By using machine learning classification model, <strong>Random Forest</strong> trained on comprehensive clinical features from the <strong>National Alzheimer's Coordinating Center (NACC)</strong> dataset, 
            the system processes demographic data, cognitive test scores (like MoCA and Functional Activities Questionnaire parameters) and medical condition to output rapid, objective diagnostic probabilities.
            This model achieves an <strong>accuracy of 86.05%</strong>, a class-discrimination <strong>AUC of 0.9534</strong> and a good <strong>specificity of 92.86%</strong> to ensure reliable clinical predictions.</p>
        <p>To ensure algorithmic transparency and clinical accountability, the platform integrates <strong>SHAP (SHapley Additive exPlanations)</strong>, a state-of-the-art <strong>Explainable AI (XAI)</strong> technique. 
            By calculating individual Shapley values for each clinical feature, the system breaks down the underlying logic behind every prediction to show how much a patient's cognitive scores, age or other factors contributed to their specific result.</p>
        <p><em>Note: This application serves as a secondary decision-support tool for academic validation and should be used in conjunction with formal clinical protocols.</em></p>
    </div>
    """, unsafe_allow_html=True)

    #how to use section
    st.markdown('##### <i class="bi bi-gear-wide-connected" style="margin-right: 8px; color: #15877B;"></i> <span style="color: #15877B;">How to Use the System</span>', unsafe_allow_html=True)
    
    use_col1, use_col2, use_col3 = st.columns(3)
    with use_col1:
        st.markdown("""
        <div class="dash-card" style="padding: 20px; min-height: 210px; border-top: 4px solid #15877B;border-bottom: 4px solid #15877B;">
            <div style="font-weight: 700; color: #15877B; font-size: 16px; margin-bottom: 10px;">
                <i class="bi bi-person-bounding-box" style="padding-right:3px;"></i> Step 1: Input Data
            </div>
            <p style="color: #4A5568; font-size: 14px; line-height: 1.5; text-align: justify;">
                Navigate to the <strong>Risk Assessment</strong> page from the sidebar menu. Choose the visit mode of the patient. Complete the demographics, lifestyle history, cognitive scores and other needed fields.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with use_col2:
        st.markdown("""
        <div class="dash-card" style="padding: 20px; min-height: 210px; border-top: 4px solid #15877B;border-bottom: 4px solid #15877B;">
            <div style="font-weight: 700; color: #15877B; font-size: 16px; margin-bottom: 10px;">
                <i class="bi bi-cpu" style="padding-right:3px;"></i> Step 2: Run Prediction
            </div>
            <p style="color: #4A5568; font-size: 14px; line-height: 1.5; text-align: justify;">
                Click the <strong>Run Prediction</strong> button to start prediction. The <strong>Random Forest</strong> model will classify the patient's cognitive stage (Normal, MCI or AD Dementia) with output probabilities in an average of 13 seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with use_col3:
        st.markdown("""
        <div class="dash-card" style="padding: 20px; min-height: 210px; border-top: 4px solid #15877B;border-bottom: 4px solid #15877B;">
            <div style="font-weight: 700; color: #15877B; font-size: 16px; margin-bottom: 10px;">
                <i class="bi bi-clipboard-data" style="padding-right:3px;"></i> Step 3: View & Export
            </div>
            <p style="color: #4A5568; font-size: 14px; line-height: 1.5; text-align: justify;">
                Review the <strong>SHAP visual charts</strong> to observe exactly which clinical features weighted the model's final decision. The <strong>Clinical PDF Report</strong> can be downloaded. All filled information, result and report will be saved in database.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    #contact section
    st.markdown('##### <i class="bi bi-envelope-paper-fill" style="margin-right: 8px; color: #15877B;"></i> <span style="color: #15877B;">Contact & Institutional Affiliation</span>', unsafe_allow_html=True)
    
    col_info, col_map = st.columns([1, 1])
    
    with col_info:

        st.markdown("""
        <div class="dash-card" style="padding: 20px; height: 100%;">
            <div style="font-weight: 700; color: #2D323E; font-size: 16px; margin-bottom: 15px;">
                Multimedia University (MMU)
            </div>
            <div style="display: flex; align-items: start; gap: 10px; margin-bottom: 12px; color: #4A5568;">
                <i class="bi bi-geo-alt-fill" style="color: #15877B; font-size: 18px;"></i>
                <span><strong>Faculty of Information Science & Technology (FIST)</strong><br>
                Jalan Ayer Keroh Lama, 75450 Bukit Beruang,<br>Melaka, Malaysia.</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px; color: #4A5568;">
                <i class="bi bi-envelope-fill" style="color: #15877B; font-size: 16px;"></i>
                <span><a href="mailto:tan.yan.san@student.mmu.edu.my" style="color: #15877B; text-decoration: none;">tan.yan.san@student.mmu.edu.my</a></span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; color: #4A5568; margin-top: 20px; padding-top: 15px; border-top: 1px solid #EEF2F6;">
                <i class="bi bi-mortarboard-fill" style="color: #15877B; font-size: 20px;"></i>
                <span style="font-size: 13px; font-weight: 500;">Final Year Project Predictive Models for Cognitive Decline Assessment Demo — 2026</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        #google map to mmu melaka
        mmu_map_html = """
        <div class="dash-card" style="overflow: hidden; height: 100%;">
            <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3986.7436995642674!2d102.27611360000002!3d2.2494934999999945!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x31d1e56b9710cf4b%3A0x66b6b12b75469278!2sMultimedia%20University!5e0!3m2!1sen!2smy!4v1779813869272!5m2!1sen!2smy" 
                width="100%" 
                height="255px" 
                style="border:0;" 
                allowfullscreen="" 
                loading="lazy" 
                referrerpolicy="no-referrer-when-downgrade">
            </iframe>
        </div>
        """
        st.markdown(mmu_map_html, unsafe_allow_html=True)

#footer------------------------------------------------------------------------
footer_html = textwrap.dedent("""
        <style>
            .clinical-footer {
                position: fixed;
                right: 0;
                bottom: 0;
                width: 100%;
                background-color: #2D323E;
                padding: 8px 30px;
                display: flex;
                text-align:right;
                justify-content:right;
            }
            .main .block-container {
                padding-bottom: 80px !important;
            }
        </style>
        
        <div class="clinical-footer">
            <div style=gap: 8px;">
                <i class="bi bi-robot" style="font-size: 16px; color: #00D2C4;margin-right:5px;"></i>
                <span style="font-size: 12px; color: #00D2C4; margin-right:3px;font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">
                    Core Engine:         
                </span>
                <span style="font-size: 12px; font-weight: 600; color: #FFFFFF;padding-right:5px; margin-right:5px; border-right: 1px solid #fff">
                    Random Forest Classifier
                </span>
                <span style="font-size: 11px; color: #A0AABF; letter-spacing: 0.3px;">
                    Trained on NACC cohort dataset &bull; MMU Final Year Project 2026
                </span>
            </div>
        </div>
    """)
st.markdown(footer_html, unsafe_allow_html=True)