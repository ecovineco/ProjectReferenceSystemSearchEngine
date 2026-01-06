# 🔎 Project Reference Search Engine

A completely offline, AI-powered semantic search tool. This application allows users to upload an Excel list of project references and search through them using natural language (concepts) rather than just exact keywords.

## 📂 Project Structure & Logic

This project consists of two main Python scripts. Here is exactly what they do:

### 1. `app.py` (The Brain)
This is the main application logic. It handles the entire lifecycle of the search engine:
* **Path Management:** Uses a helper function (`get_base_path`) to determine if it's running as a script or a compiled `.exe`, ensuring it always finds its files.
* **Data Ingestion:**
    1.  Reads your Excel file.
    2.  Extracts specific columns: `ProjectID`, `Name`, `Description`, `RefStartDate`, `RefEndDate`, `TotalContractValue`, `EcorysContractValue`.
    3.  **Vectorization:** It combines the Name (weighted 3x) and Description into a single text block and uses the `all-mpnet-base-v2` AI model to convert them into mathematical vectors (embeddings).
    4.  **Caching:** Saves these vectors to a local file (`embeddings.pkl`) so the "Brain" doesn't need to be rebuilt every time.
* **Search Engine:**
    1.  Takes the user's search query.
    2.  Converts the query into a vector using the same AI model.
    3.  Performs a **Semantic Similarity Search** (Cosine Similarity) to find projects that match the *meaning* of the query.
    4.  Formats and displays the top results, including cleaning up date formats.

### 2. `run_app.py` (The Launcher)
This is a wrapper script required **only** for creating the Windows Executable.
* **Why is it needed?** You cannot run `streamlit` directly inside a standard `.exe`.
* **What it does:** It creates a "fake" command-line environment inside Python that executes `streamlit run app.py` programmatically. It acts as the bridge between the Windows OS and the Streamlit framework.

---

## ⚙️ Installation

**Requirements:**
* Python 3.8 or higher

**Install Dependencies:**
Open your terminal and run:
```bash
pip install streamlit pandas sentence-transformers openpyxl numpy pyinstaller

## 🚀 How to Run

You have two ways to use this application:

### Option A: Run as a Python Script (For Development)
If you have Python installed and want to use or edit the code:

1.  Open your terminal in the project folder.
2.  Run the command:
    ```bash
    streamlit run app.py
    ```
3.  The app will open in your browser at `http://localhost:8501`.

### Option B: Create a Windows App (For Distribution)
To bundle this into a folder that runs on any Windows computer (without needing Python installed), follow these steps:

1.  **Prime the Model:** Run the app once (Option A) and build the index to ensure the `model_cache` folder is created on your machine.
2.  **Build the Exe:** Run this command in your terminal:
    ```bash
    pyinstaller --noconfirm --onedir --windowed --name "ProjectSearchApp" --clean --collect-all streamlit --collect-all sentence_transformers --collect-all torch --copy-metadata streamlit --copy-metadata tqdm --copy-metadata regex --copy-metadata requests --copy-metadata packaging --copy-metadata filelock --copy-metadata numpy --copy-metadata tokenizers --add-data "app.py;." --add-data "model_cache;model_cache" run_app.py
    ```
    > **Note:** We use `--onedir` instead of `--onefile` because it makes the app start up much faster.

3.  **Locate the App:**
    * Go to the new `dist` folder.
    * Open `ProjectSearchApp`.
    * Run `ProjectSearchApp.exe`.

---

## 📋 Input Data Format

The application expects an Excel file (`.xlsx` or `.xlsm`) with the following headers:

| Column Name | Description |
| :--- | :--- |
| **ProjectID** | Unique ID (e.g., A-100) |
| **Name** | Title of the project |
| **Description** | Full text description |
| **RefStartDate** | Start Date |
| **RefEndDate** | End Date |
| **TotalContractValue** | Total value |
| **EcorysContractValue** | Relevant share value |

---

## 🛠 Troubleshooting

* **App closes immediately:** If the `.exe` closes instantly, it likely means a library is missing. Try building with `--console` instead of `--windowed` to see the error message.
* **Search Results are empty:** Ensure you have clicked the **"Build/Update Database"** button in the sidebar after uploading a new Excel file.