# 🔎 Project Reference Search Engine

An offline, AI-powered semantic search tool for project references. Upload an Excel list of projects and search through them using natural language — find projects by concept and meaning, not just exact keywords.

---

## 📂 Project Structure

Everything lives in a **single file**:

```
app.py              ← The entire application (launcher + Streamlit UI + search logic)
data/
  embeddings/
    embeddings.pkl  ← Auto-generated search index (created on first database build)
  model/            ← Auto-downloaded AI model (downloaded on first run)
```

The `data/` folder and its subdirectories are created automatically on first launch — you don't need to set them up manually.

---

## ⚙️ How It Works

### Self-Launching

`app.py` contains a built-in launcher at the top. When you run `python app.py`, it detects that Streamlit is not yet running and programmatically calls `streamlit run app.py` for you. This means you never need to type the `streamlit run` command yourself.

### Data Ingestion Pipeline

When you upload an Excel file and click **"Build/Update Database"**, the app:

1. **Reads** the Excel file with `pandas`.
2. **Fuzzy-matches columns** — your column headers don't need to be exact. `ProjectID`, `project_code`, `Project Code`, and `proj id` all resolve to the same canonical field. The same flexibility applies to all columns (see the full alias list in the Column Mapping section below).
3. **Combines text** for each project: `ProjectName` is repeated 3× concatenated with `Description`. This weighting makes the project name 3× more influential than the description during search.
4. **Generates embeddings** using the `all-mpnet-base-v2` sentence-transformer model. Each project becomes a high-dimensional vector capturing its semantic meaning.
5. **Saves the index** to `data/embeddings/embeddings.pkl` so it only needs to be rebuilt when your data changes.

### Search Engine

Three search modes are available:

| Mode | How it works | Best for |
|---|---|---|
| **Semantic (AI-based)** | Encodes your query into a vector and finds projects with the most similar meaning using cosine similarity | Exploratory searches, concepts, synonyms |
| **Keyword (exact match)** | Counts how many of your query words appear in each project's combined text | Searching specific names, IDs, or exact terms |
| **Hybrid (both combined)** | Merges semantic score (70%) + keyword score (30%) | General-purpose searches |

### AI Model

The app uses `sentence-transformers/all-mpnet-base-v2` from Hugging Face. On first run it downloads the model into `data/model/`. On all subsequent runs it loads from disk with no network access needed.

---

## 📋 Input Data Format

The app expects an Excel file (`.xlsx` or `.xlsm`). The three required columns are **Project Code**, **Project Name**, and **Description**. The rest are optional.

### Column Mapping

Column names are matched **case-insensitively** and **flexibly**. Any of these variations will be recognised:

| Canonical Field | Accepted variations |
|---|---|
| `ProjectCode` | `project code`, `projectcode`, `project_code`, `projectid`, `project_id`, `project id`, `id`, `proj id`, `projid` |
| `ProjectName` | `project name`, `projectname`, `project_name`, `name`, `title`, `project title` |
| `Description` | `description`, `desc`, `project description`, `projectdescription`, `summary` |
| `StartDate` | `start date`, `startdate`, `start_date`, `refstartdate`, `ref start date`, `begin date`, `from` |
| `EndDate` | `end date`, `enddate`, `end_date`, `refenddate`, `ref end date`, `finish date`, `to` |
| `TotalContractValue` | `total contract value`, `totalcontractvalue`, `total_contract_value`, `total value`, `contract value` |
| `EcorysContractValue` | `ecorys contract value`, `ecoryscontractvalue`, `ecorys_contract_value`, `ecorys value` |

If a required column can't be matched, the app shows an error listing exactly which columns were found and what it was looking for.

---

## 🚀 Installation & Running

### Requirements

- Python 3.8 or higher
- Internet access on first run only (to download the AI model, ~420 MB)

### Install Dependencies

```bash
pip install streamlit pandas sentence-transformers openpyxl numpy python-dateutil huggingface_hub
```

### Run the App

```bash
python app.py
```

The browser will open automatically at `http://localhost:8501`.

> **No need to run `streamlit run app.py`** — the built-in launcher handles this.

---

## 🖥️ Using the App

### Step 1 — Ingest Your Data (Sidebar)

1. Open the **sidebar** on the left.
2. Upload your Excel file under **"1. Data Ingestion"**.
3. Click **"Build/Update Database"**.
4. Wait for the success message. The database status panel below shows how many projects are indexed and when it was last updated.
5. Optionally use **"Browse Database"** to inspect the loaded data.

> You only need to do this once, or whenever your Excel file changes. The index persists on disk.

### Step 2 — Search

1. Type your query in the search box (e.g., *"sustainable energy policy in urban areas"*).
2. Select a **Search mode** (Hybrid is recommended for general use).
3. Press **Enter** or click **"Run Search"**.

### Advanced Options

- **Minimum relevance (%)** — filters out weak matches. Lower = more results, higher = only strong matches. Default is 25%.
- **Maximum number of results** — how many candidates to retrieve before filtering. Default is 20.

### Filters

All filters are optional and can be combined:

- **Starting / Ending year** — multi-select from available years in the database.
- **Total Contract Value / Ecorys Contract Value** — slider range filters.
- **Project Code (contains)** — text filter, e.g. type `2024` to show only projects with that in their code.
- **Sort by** — relevance, start/end date, contract value, or project name.
- Use **"Reset Filters"** to clear all active filters at once.

### Results Table

Results show: Project Code, Relevance %, Start Date, End Date, Total Contract Value, Ecorys Contract Value, Project Name, and Description. Columns that don't exist in your data are hidden automatically.

---

## 📦 Building a Windows Executable

To distribute the app as a standalone `.exe` folder (no Python installation required):

**1. Prime the model first** — run the app once and build the database so the model is downloaded into `data/model/`.

**2. Install PyInstaller:**
```bash
pip install pyinstaller
```

**3. Build:**
```bash
pyinstaller --noconfirm --onedir --windowed --name "ProjectSearchApp" --clean \
  --collect-all streamlit \
  --collect-all sentence_transformers \
  --collect-all torch \
  --copy-metadata streamlit \
  --copy-metadata tqdm \
  --copy-metadata regex \
  --copy-metadata requests \
  --copy-metadata packaging \
  --copy-metadata filelock \
  --copy-metadata numpy \
  --copy-metadata tokenizers \
  --add-data "app.py;." \
  --add-data "data/model;data/model" \
  app.py
```

> Use `--onedir` (not `--onefile`) — it starts significantly faster.

**4. Distribute** the `dist/ProjectSearchApp/` folder. Run `ProjectSearchApp.exe` inside it.

> **Troubleshooting:** If the `.exe` closes immediately, replace `--windowed` with `--console` to see the error output in a terminal window.

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| Search returns no results | Check that you've built the database first (sidebar). Lower the **Minimum relevance** threshold in Advanced Options. |
| Column not recognised | Check the Column Mapping table above. The app will tell you exactly which columns it found vs expected. |
| Model download fails | Ensure internet access on first run. The model (~420 MB) is saved to `data/model/` and never re-downloaded. |
| Date columns show raw values | Dates in almost any format are handled automatically (`10 Aug 2019`, `10/08/2019`, `2019-08-10`, etc.). |
| `.exe` closes immediately | Build with `--console` instead of `--windowed` to read the error message. |
