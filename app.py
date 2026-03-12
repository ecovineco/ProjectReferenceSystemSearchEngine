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
import re

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

# --- HELPER: Fuzzy column matching ---
# Maps canonical column names to common variations found in Excel files.
# Matching is case-insensitive and ignores extra spaces.
COLUMN_ALIASES = {
    'ProjectCode':         ['project code', 'projectcode', 'project_code', 'projectid', 'project_id', 'project id', 'id', 'proj id', 'projid'],
    'ProjectName':         ['project name', 'projectname', 'project_name', 'name', 'title', 'project title'],
    'Description':         ['description', 'desc', 'project description', 'projectdescription', 'summary'],
    'StartDate':           ['start date', 'startdate', 'start_date', 'refstartdate', 'ref start date', 'ref_start_date', 'begin date', 'from'],
    'EndDate':             ['end date', 'enddate', 'end_date', 'refenddate', 'ref end date', 'ref_end_date', 'finish date', 'to'],
    'TotalContractValue':  ['total contract value', 'totalcontractvalue', 'total_contract_value', 'total value', 'totalvalue', 'contract value'],
    'EcorysContractValue': ['ecorys contract value', 'ecoryscontractvalue', 'ecorys_contract_value', 'ecorys value', 'ecorysvalue'],
}

def match_columns(df_columns):
    """
    Attempts to map the actual Excel columns to our canonical names.
    Returns a dict {canonical_name: actual_column_name} for matches found,
    and a list of canonical names that could not be matched.
    """
    # Normalize: strip whitespace, lowercase
    normalized = {col: re.sub(r'\s+', ' ', col.strip().lower()) for col in df_columns}

    matched = {}
    unmatched = []

    for canonical, aliases in COLUMN_ALIASES.items():
        found = False
        for actual_col, norm_col in normalized.items():
            if norm_col in aliases or norm_col == canonical.lower():
                matched[canonical] = actual_col
                found = True
                break
        if not found:
            unmatched.append(canonical)

    return matched, unmatched

# --- HELPER: Date formatting (defined once, not inside loops) ---
def format_date(val):
    """Formats a date value to YYYY-MM-DD string, removing time component.
    Handles formats like '10 Aug 2019', '10/08/2019', '2019-08-10 00:00:00.000', etc.
    """
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    if pd.isna(val) or str(val).strip() == "":
        return ""
    try:
        from dateutil import parser as dateutil_parser
        return str(dateutil_parser.parse(str(val), dayfirst=True).date())
    except Exception:
        try:
            return str(pd.to_datetime(val, dayfirst=True).date())
        except Exception:
            return str(val)

# --- HELPER: Smart model loading ---
def _model_is_cached():
    """Check if the model files already exist locally (avoid network call)."""
    # A valid SentenceTransformer directory must have config.json and pytorch_model.bin or model.safetensors
    config_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
    weights_exist = (
        os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or
        os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))
    )
    return config_exists and weights_exist

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load the sentence-transformer model.
    Downloads to data/model/ only on first run; subsequent loads skip the network check.
    """
    if not _model_is_cached():
        snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR)
    return SentenceTransformer(MODEL_DIR)


# --- 2. APP UI SETUP ---
st.set_page_config(page_title="Project Reference System", layout="wide")
st.title("🔎 Project Reference System")

# --- 3. SIDEBAR: DATA INGESTION ---
st.sidebar.header("1. Data Ingestion")
st.sidebar.info(
    "**Expected columns:**\n"
    "- **Project Code** *(required)*\n"
    "- **Project Name** *(required)*\n"
    "- **Description** *(required)*\n"
    "- **Start Date** *(optional)*\n"
    "- **End Date** *(optional)*\n"
    "- **Total Contract Value** *(optional)*\n"
    "- **Ecorys Contract Value** *(optional)*\n\n"
    "Column names are matched flexibly (e.g. 'project_code', 'ProjectID', 'Project Code' all work)."
)
uploaded_file = st.sidebar.file_uploader("Upload your Excel Data", type=['xlsx', 'xlsm'])

if st.sidebar.button("Build/Update Database"):
    if uploaded_file:
        with st.spinner("Reading Excel and creating Search Engine (may take some time)"):
            try:
                # A. Read Excel
                df = pd.read_excel(uploaded_file)

                # B. Fuzzy column matching
                col_map, missing = match_columns(df.columns)

                # ProjectCode, ProjectName, Description are required; the rest are optional
                required_canonical = ['ProjectCode', 'ProjectName', 'Description']
                missing_required = [c for c in required_canonical if c not in col_map]

                if missing_required:
                    st.sidebar.error(
                        f"Could not find required columns: **{', '.join(missing_required)}**.\n\n"
                        f"Columns found in your file: {list(df.columns)}\n\n"
                        f"Accepted variations: " +
                        ", ".join(f"'{c}' → {COLUMN_ALIASES[c]}" for c in missing_required)
                    )
                else:
                    # C. Rename matched columns to canonical names & keep only those
                    reverse_map = {actual: canonical for canonical, actual in col_map.items()}
                    df = df.rename(columns=reverse_map)
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]
                    available_cols = [c for c in COLUMN_ALIASES.keys() if c in df.columns]
                    df = df[available_cols]

                    if missing:
                        st.sidebar.warning(
                            f"Optional columns not found (will be empty): {', '.join(missing)}"
                        )

                    # D. Combine Text for embeddings
                    # Weighted: Repeat the ProjectName 3 times to make it 3x more important than the description
                    df['combined_text'] = (df['ProjectName'].fillna('') + ". ") * 3 + df['Description'].fillna('')

                    # E. Load Model
                    model = load_model()

                    # F. Create Embeddings
                    corpus_embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

                    # G. Save Index to Disk (full replace)
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump({'embeddings': corpus_embeddings, 'data': df}, f)

                    mod_time = os.path.getmtime(CACHE_FILE)
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))

                    st.sidebar.success(f"Success! Database was updated at: **{timestamp}** ({len(df)} projects indexed)")

            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")
    else:
        st.sidebar.warning("Please upload an Excel file first.")

# --- SIDEBAR: Database status indicator ---
st.sidebar.divider()
st.sidebar.header("Database Status")
if os.path.exists(CACHE_FILE):
    mod_time = os.path.getmtime(CACHE_FILE)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
    try:
        with open(CACHE_FILE, "rb") as f:
            cache_peek = pickle.load(f)
            num_records = len(cache_peek['data'])
        st.sidebar.success(f"Database loaded: **{num_records}** projects\n\nLast updated: **{timestamp}**")

        # Browse database expander
        with st.sidebar.expander("Browse Database"):
            browse_df = cache_peek['data'].drop(columns=['combined_text'], errors='ignore')
            browse_df = browse_df.loc[:, ~browse_df.columns.duplicated(keep='first')]

            # Format date columns for display
            for date_col in ['StartDate', 'EndDate']:
                if date_col in browse_df.columns:
                    browse_df[date_col] = browse_df[date_col].apply(format_date)

            # Rename columns for readability
            display_names = {
                'ProjectCode': 'Project Code',
                'ProjectName': 'Project Name',
                'StartDate': 'Start Date',
                'EndDate': 'End Date',
                'TotalContractValue': 'Total Contract Value',
                'EcorysContractValue': 'Ecorys Contract Value',
            }
            browse_df = browse_df.rename(columns={k: v for k, v in display_names.items() if k in browse_df.columns})

            st.dataframe(browse_df, use_container_width=True, hide_index=True, height=400)

    except Exception:
        st.sidebar.warning("Database file exists but could not be read.")
else:
    st.sidebar.warning("No database found. Upload an Excel file and click 'Build/Update Database' to get started.")

# --- 4. MAIN AREA: SEARCH INTERFACE ---
st.header("2. Search Engine")

# --- Search input row ---
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter search term:", placeholder="e.g., Sustainable energy in urban areas...")
with col2:
    st.write("")
    st.write("")
    search_clicked = st.button("Run Search", use_container_width=True)

# --- Search mode selection ---
search_mode = st.radio(
    "Search mode:",
    ["Semantic (AI-based)", "Keyword (exact match)", "Hybrid (both combined)"],
    horizontal=True,
    help=(
        "**Semantic**: Finds projects with similar *meaning*, even if exact words differ. "
        "Best for exploratory searches like 'renewable energy policy'.\n\n"
        "**Keyword**: Finds projects containing the exact words you typed. "
        "Best for searching specific names, IDs, or precise terms.\n\n"
        "**Hybrid**: Combines both methods — semantic similarity and keyword matching — "
        "and merges the scores. Best general-purpose option."
    )
)

# --- Advanced Options (collapsible) ---
with st.expander("Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        relevance_threshold = st.slider(
            "Minimum relevance (%)",
            min_value=0, max_value=100, value=25, step=5,
            help="Only show results above this relevance score. "
                 "Lower values show more results (including weaker matches); "
                 "higher values show only strong matches."
        )
    with adv_col2:
        top_k = st.slider(
            "Maximum number of results",
            min_value=5, max_value=100, value=20, step=5,
            help="How many candidate results to retrieve before filtering. "
                 "Increase this if you want to see more results."
        )

# --- Filters ---
# Pre-load database metadata for dynamic filter options
_filter_start_years = []
_filter_end_years = []
_filter_total_value_range = (0, 0)
_filter_ecorys_value_range = (0, 0)

if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            _filter_cache = pickle.load(f)
            _filter_df = _filter_cache['data']

        # Extract available start years
        if 'StartDate' in _filter_df.columns:
            _start_dates = _filter_df['StartDate'].apply(
                lambda v: format_date(v) if not (pd.isna(v) if not isinstance(v, str) else v.strip() == "") else ""
            )
            _filter_start_years = sorted(set(
                int(d[:4]) for d in _start_dates if d and len(d) >= 4 and d[:4].isdigit()
            ))

        # Extract available end years
        if 'EndDate' in _filter_df.columns:
            _end_dates = _filter_df['EndDate'].apply(
                lambda v: format_date(v) if not (pd.isna(v) if not isinstance(v, str) else v.strip() == "") else ""
            )
            _filter_end_years = sorted(set(
                int(d[:4]) for d in _end_dates if d and len(d) >= 4 and d[:4].isdigit()
            ))

        # Extract total contract value range
        if 'TotalContractValue' in _filter_df.columns:
            _tcv = pd.to_numeric(_filter_df['TotalContractValue'], errors='coerce').dropna()
            if not _tcv.empty:
                _filter_total_value_range = (int(_tcv.min()), int(_tcv.max()))

        # Extract ecorys contract value range
        if 'EcorysContractValue' in _filter_df.columns:
            _ecv = pd.to_numeric(_filter_df['EcorysContractValue'], errors='coerce').dropna()
            if not _ecv.empty:
                _filter_ecorys_value_range = (int(_ecv.min()), int(_ecv.max()))

        del _filter_cache, _filter_df
    except Exception:
        pass

# Reset filters button: clears all filter session state keys
if 'reset_filters' not in st.session_state:
    st.session_state['reset_filters'] = False

with st.expander("Filters"):
    if st.button("Reset Filters"):
        for key in ['filter_start_years', 'filter_end_years', 'filter_total_value',
                     'filter_ecorys_value', 'filter_project_code', 'filter_sort_by']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_start_years = st.multiselect(
            "Starting year(s)",
            options=_filter_start_years,
            default=st.session_state.get('filter_start_years', []),
            key='filter_start_years',
            help="Only show projects that started in one of the selected years."
        )
        filter_end_years = st.multiselect(
            "Ending year(s)",
            options=_filter_end_years,
            default=st.session_state.get('filter_end_years', []),
            key='filter_end_years',
            help="Only show projects that ended in one of the selected years."
        )
    with filter_col2:
        if _filter_total_value_range[0] < _filter_total_value_range[1]:
            filter_total_value = st.slider(
                "Total Contract Value (€)",
                min_value=_filter_total_value_range[0],
                max_value=_filter_total_value_range[1],
                value=_filter_total_value_range,
                key='filter_total_value',
                help="Only show projects within this total contract value range."
            )
        else:
            filter_total_value = None

        if _filter_ecorys_value_range[0] < _filter_ecorys_value_range[1]:
            filter_ecorys_value = st.slider(
                "Ecorys Contract Value (€)",
                min_value=_filter_ecorys_value_range[0],
                max_value=_filter_ecorys_value_range[1],
                value=_filter_ecorys_value_range,
                key='filter_ecorys_value',
                help="Only show projects within this Ecorys contract value range."
            )
        else:
            filter_ecorys_value = None

    filter_project_code = st.text_input("Filter by Project Code (contains):", placeholder="e.g., 2024",
                                        key='filter_project_code',
                                        help="Only show projects whose code contains this text.")

    sort_by = st.selectbox(
        "Sort results by:",
        ["Relevance (highest first)", "Start Date (newest first)", "Start Date (oldest first)",
         "Total Value (highest first)", "Total Value (lowest first)", "Project Name (A-Z)"],
        key='filter_sort_by',
        help="Choose how to order the search results."
    )

# --- SEARCH LOGIC ---
def keyword_search(query_text, df):
    """Simple keyword search: scores each row by how many query words appear in its combined text."""
    query_words = query_text.lower().split()
    scores = []
    for _, row in df.iterrows():
        text = str(row.get('combined_text', '')).lower()
        if not query_words:
            scores.append(0.0)
            continue
        matches = sum(1 for w in query_words if w in text)
        scores.append(matches / len(query_words))
    return scores


# Trigger search on Enter key (query exists) or Button click
if search_clicked or query:
    if not query:
        st.warning("Please enter a query.")
    elif not os.path.exists(CACHE_FILE):
        st.error("Index not found! Please upload data and click 'Build/Update Database' in the sidebar.")
    else:
        try:
            # A. Load Index
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
                corpus_embeddings = cache['embeddings']
                df = cache['data']
                df = df.loc[:, ~df.columns.duplicated(keep='first')]

            # B. Compute scores based on search mode
            if search_mode == "Keyword (exact match)":
                # Pure keyword search
                kw_scores = keyword_search(query, df)
                scored_indices = [(i, score) for i, score in enumerate(kw_scores) if score > 0]
                scored_indices.sort(key=lambda x: x[1], reverse=True)
                scored_indices = scored_indices[:top_k]
                # Convert to percentage
                hits_with_scores = [(idx, round(score * 100, 1)) for idx, score in scored_indices]

            elif search_mode == "Semantic (AI-based)":
                # Pure semantic search
                model = load_model()
                query_embedding = model.encode(query, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
                hits_with_scores = [(hit['corpus_id'], round(hit['score'] * 100, 1)) for hit in hits[0]]

            else:
                # Hybrid: combine semantic + keyword scores
                model = load_model()
                query_embedding = model.encode(query, convert_to_tensor=True)
                hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)

                kw_scores = keyword_search(query, df)

                # Merge: 70% semantic + 30% keyword
                hybrid_results = []
                seen_ids = set()
                for hit in hits[0]:
                    idx = hit['corpus_id']
                    semantic_score = hit['score'] * 100
                    kw_score = kw_scores[idx] * 100
                    combined = 0.7 * semantic_score + 0.3 * kw_score
                    hybrid_results.append((idx, round(combined, 1)))
                    seen_ids.add(idx)

                # Also include any high keyword matches not in semantic results
                for idx, kw_s in enumerate(kw_scores):
                    if idx not in seen_ids and kw_s > 0.3:
                        combined = 0.3 * kw_s * 100
                        hybrid_results.append((idx, round(combined, 1)))

                hybrid_results.sort(key=lambda x: x[1], reverse=True)
                hits_with_scores = hybrid_results[:top_k]

            # C. Format Results
            results = []
            for row_idx, score in hits_with_scores:
                # Apply relevance threshold
                if score <= relevance_threshold:
                    continue

                r = df.iloc[row_idx]

                result_item = {
                    "Relevance": f"{score}%",
                    "_relevance_num": score,
                    "Project Name": r.get('ProjectName', ''),
                    "Description": r.get('Description', ''),
                    "Start Date": format_date(r.get('StartDate')),
                    "End Date": format_date(r.get('EndDate')),
                    "Total Contract Value": r.get('TotalContractValue'),
                    "Ecorys Contract Value": r.get('EcorysContractValue')
                }

                if 'ProjectCode' in r:
                    result_item["Project Code"] = r['ProjectCode']

                results.append(result_item)

            # D. Apply metadata filters
            if results:
                results_df = pd.DataFrame(results)

                # Filter by Project Code
                if filter_project_code:
                    results_df = results_df[
                        results_df['Project Code'].astype(str).str.contains(filter_project_code, case=False, na=False)
                    ]

                # Filter by start year(s)
                if filter_start_years:
                    results_df = results_df[
                        results_df['Start Date'].apply(
                            lambda x: int(x[:4]) in filter_start_years if x and len(x) >= 4 and x[:4].isdigit() else False
                        )
                    ]

                # Filter by end year(s)
                if filter_end_years:
                    results_df = results_df[
                        results_df['End Date'].apply(
                            lambda x: int(x[:4]) in filter_end_years if x and len(x) >= 4 and x[:4].isdigit() else False
                        )
                    ]

                # Filter by total contract value range
                if filter_total_value is not None and filter_total_value != _filter_total_value_range:
                    _tcv_col = pd.to_numeric(results_df['Total Contract Value'], errors='coerce').fillna(0)
                    results_df = results_df[
                        (_tcv_col >= filter_total_value[0]) & (_tcv_col <= filter_total_value[1])
                    ]

                # Filter by ecorys contract value range
                if filter_ecorys_value is not None and filter_ecorys_value != _filter_ecorys_value_range:
                    _ecv_col = pd.to_numeric(results_df['Ecorys Contract Value'], errors='coerce').fillna(0)
                    results_df = results_df[
                        (_ecv_col >= filter_ecorys_value[0]) & (_ecv_col <= filter_ecorys_value[1])
                    ]

                # E. Apply sorting
                if sort_by == "Start Date (newest first)":
                    results_df = results_df.sort_values('Start Date', ascending=False)
                elif sort_by == "Start Date (oldest first)":
                    results_df = results_df.sort_values('Start Date', ascending=True)
                elif sort_by == "Total Value (highest first)":
                    results_df['_sort_val'] = pd.to_numeric(results_df['Total Contract Value'], errors='coerce').fillna(0)
                    results_df = results_df.sort_values('_sort_val', ascending=False)
                    results_df = results_df.drop(columns=['_sort_val'])
                elif sort_by == "Total Value (lowest first)":
                    results_df['_sort_val'] = pd.to_numeric(results_df['Total Contract Value'], errors='coerce').fillna(0)
                    results_df = results_df.sort_values('_sort_val', ascending=True)
                    results_df = results_df.drop(columns=['_sort_val'])
                elif sort_by == "Project Name (A-Z)":
                    results_df = results_df.sort_values('Project Name', ascending=True)
                # Default: Relevance (already sorted by score)

                # Drop internal helper column
                results_df = results_df.drop(columns=['_relevance_num'], errors='ignore')

                # F. Display Results
                if not results_df.empty:
                    st.success(f"Found {len(results_df)} matches.")

                    # Reorder columns
                    cols = [
                        'Relevance',
                        'Start Date', 'End Date',
                        'Total Contract Value', 'Ecorys Contract Value',
                        'Project Name', 'Description'
                    ]
                    if 'Project Code' in results_df.columns:
                        cols.insert(0, 'Project Code')
                    # Only include columns that exist
                    cols = [c for c in cols if c in results_df.columns]
                    results_df = results_df[cols]

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No matches after applying filters. Try relaxing your filters.")
            else:
                st.info("No relevant matches found. Try different keywords or lower the relevance threshold in Advanced Options.")

        except Exception as e:
            st.error(f"Error during search: {e}")
