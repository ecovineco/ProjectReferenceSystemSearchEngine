import os
import sys

# --- 0. LAUNCHER (Merged from run_app.py) ---
# When run directly with "python app.py", launch Streamlit to serve this file.
# When Streamlit re-executes this file, its runtime already exists, so we skip.
def _is_streamlit_running():
    try:
        from streamlit.runtime import exists
        return exists()
    except ImportError:
        return False

if not _is_streamlit_running():
    import streamlit.web.cli as stcli

    def resolve_path(path):
        if getattr(sys, "frozen", False):
            basedir = sys._MEIPASS
        else:
            basedir = os.path.dirname(__file__)
        return os.path.join(basedir, path)

    app_path = resolve_path("app.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false"
    ]
    sys.exit(stcli.main())

# --- Beyond this point: Streamlit app code ---
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
import pickle
import time

# --- 1. PATH CONFIGURATION (Crucial for .exe) ---
def get_base_path():
    """ 
    Determines the correct root directory.
    - If running as a Python script: uses the file's folder.
    - If running as a PyInstaller .exe: uses the temporary sys._MEIPASS folder.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
DATA_DIR = os.path.join(BASE_DIR, "data")

# Define paths: data/embeddings/ for index, data/model/ for the AI model cache
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODEL_DIR = os.path.join(DATA_DIR, "model")
CACHE_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.pkl")
MODEL_REPO = 'sentence-transformers/all-mpnet-base-v2'

# Auto-create the data directories if they don't exist yet
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_model():
    """Download model directly into data/model/ and load from there.
    Uses local_dir (not cache_dir) to avoid symlinks, which break on Windows/OneDrive.
    """
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    return SentenceTransformer(MODEL_DIR)

# --- 2. APP UI SETUP ---
st.set_page_config(page_title="Project Reference System", layout="wide")
st.title("🔎 Project Reference System")

# --- 3. SIDEBAR: DATA INGESTION ---
st.sidebar.header("1. Data Ingestion")
st.sidebar.info(f"Upload the Project Reference List in an Excel File containing ProjectID, Name and Description columns")
uploaded_file = st.sidebar.file_uploader("Upload your Excel Data", type=['xlsx', 'xlsm'])

if st.sidebar.button("Build/Update Database"):
    if uploaded_file:
        with st.spinner("Reading Excel and creating Search Engine (may take some time)"):
            try:
                # A. Read Excel (Reads the first visible sheet by default)
                # You can change to sheet_name='Data' if strictly required
                df = pd.read_excel(uploaded_file)
                
                # B. Validation
                # Define all columns we want to keep
                required_cols = [
                    'ProjectID', 'Name', 'Description', 
                    'RefStartDate', 'RefEndDate', 
                    'TotalContractValue', 'EcorysContractValue'
                ]
                if not all(col in df.columns for col in required_cols):
                    st.sidebar.error(f"Error: Excel is missing columns. Found: {list(df.columns)}")
                else:
                    # C. Combine Text
                    df = df[required_cols]
                    # Weighted: Repeat the Name 3 times to make it 3x more important than the description
                    df['combined_text'] = (df['Name'].fillna('') + ". ") * 3 + df['Description'].fillna('')
                    
                    # D. Load Model (Downloads to data/model/ on first run)
                    model = load_model()
                    
                    # E. Create Embeddings
                    corpus_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)
                    
                    # F. Save Index to Disk
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump({'embeddings': corpus_embeddings, 'data': df}, f)
                    
                    mod_time = os.path.getmtime(CACHE_FILE)
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))

                    st.sidebar.success(f"Success! Database was updated at: **{timestamp}**")
                    
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
    else:
        st.sidebar.warning("Please upload an Excel file first.")

# --- 4. MAIN AREA: SEARCH INTERFACE ---
st.header("2. Search Engine")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter search term:", placeholder="e.g., Sustainable energy in urban areas...")
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    search_clicked = st.button("Run Search", use_container_width=True)

# Trigger search on Enter key (query exists) or Button click
if search_clicked or query:
    if not query:
        st.warning("Please enter a query.")
    elif not os.path.exists(CACHE_FILE):
        st.error("Index not found! Please upload data and click 'Build Index' in the sdebar.")
    else:
        try:

            # A. Load Index
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                corpus_embeddings = cache['embeddings']
                df = cache['data']
            
            # B. Load Model (Fast, from local cache)
            model = load_model()
            
            # C. Encode Query & Search
            query_embedding = model.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=20)
            
            # D. Format Results
            results = []
            for hit in hits[0]:
                score = round(hit['score'] * 100, 1)
                
                # Filter: Only show results with >25% relevance
                if score > 25: 
                    row_idx = hit['corpus_id']
                    r = df.iloc[row_idx]
                    
                    # Helper function to format dates (removes the hour/time)
                    def format_date(val):
                        if pd.isna(val) or str(val).strip() == "":
                            return ""
                        try:
                            return str(pd.to_datetime(val).date())
                        except:
                            return str(val) # Fallback if not a date

                    # Build result object
                    result_item = {
                        "Relevance": f"{score}%",
                        "Project Name": r['Name'],
                        "Description": r['Description'],
                        "Start Date": format_date(r.get('RefStartDate')),
                        "End Date": format_date(r.get('RefEndDate')),
                        "Total Value": r.get('TotalContractValue'),
                        "Ecorys Value": r.get('EcorysContractValue')
                    }
                    
                    # Capture ProjectID safely
                    if 'ProjectID' in r:
                        result_item["ProjectID"] = r['ProjectID']
                        
                    results.append(result_item)            
            # E. Display Results
            if results:
                st.success(f"Found {len(results)} matches.")
                
                # Create nice dataframe for display
                results_df = pd.DataFrame(results)
                
                # Reorder columns to put Relevance first, Name second
                cols = [
                    'Relevance', 
                    'Start Date', 'End Date', 
                    'Total Value', 'Ecorys Value',
                    'Project Name', 'Description'
                ]
                # --- CHANGE: Put ProjectID first if it exists ---
                if 'ProjectID' in results_df.columns:
                    cols.insert(0, 'ProjectID')
                results_df = results_df[cols]
                
                st.dataframe(
                    results_df, 
                    use_container_width=True, 
                    hide_index=True
                )
            else:
                st.info("No relevant matches found. Try different keywords.")
                
        except Exception as e:
            st.error(f"Error during search: {e}")