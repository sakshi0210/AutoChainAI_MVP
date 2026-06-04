# AutoChain AI 

### Agentic Supply Chain Intelligence & Predictive Planning for Automotive Manufacturers

AutoChain AI is a proactive, AI-powered command center designed to monitor supplier operational risks, forecast inventory levels and stockout windows using Machine Learning, compare nearshore supplier alternatives, and extract structured action items from team meeting transcript syncs.



## The Problem
Modern automotive manufacturing operates on highly sensitive "Just-In-Time" (JIT) supply chains. Disruptions such as **supplier failures, weather events, port congestions, geopolitical tariffs, and commodity price spikes** can halt assembly lines, costing manufacturers millions of dollars. 

Traditional **ERP systems** act as passive ledgers—they report supply bottlenecks only **after** the shortage has already occurred. 

## The Solution
**AutoChain AI** sits as an active intelligence layer on top of operational data. It continuously predicts supplier failures, forecasts stock depletion rates, provides alternate sourcing trade-offs, and digests unstructured planning data into immediate procurement checklists.

---

## Core Features

1. **Executive Operations Command**:
   - Dynamic global sourcing map displaying regional risk concentrations.
   - Interactive, glowing KPI metrics indicating average risk, active alerts, and stockout windows.
   - Real-time automated system operations logs.
2. **Supplier Risk & Nearshore Sourcing**:
   - Multi-factor risk engine computing logistics delays, price volatility, climate risks, and geopolitical indices.
   - Dynamic gauge scorecards showing risk levels from Stable to Critical.
   - Nearshore alternative sourcing recommendation matrix comparing Lead Times, Cost Differentials, and Pros/Cons.
3. **ML Inventory Forecasting & Commodities**:
   - Continuous time-series prediction utilizing a **RandomForestRegressor** trained on seasonal demand patterns.
   - High-fidelity Plotly charts mapping inventory timelines against safety stock thresholds to predict exact stockout dates and recommend reorder times.
   - Shaded uncertainty-bound price forecasts for raw battery assets: **Lithium, Nickel, Copper, and Steel**.
4. **AI Meeting Action Extractor**:
   - Ingests unstructured voice-to-text transcripts from daily syncs.
   - Leverages **Google Gemini 1.5 Flash** to extract tasks, assignees, deadlines, and priorities into interactive todo checklists.
   - Employs a robust local NLP regex parser fallback to ensure functionality even without internet/API keys.

---

## Tech Stack & Architecture

- **Frontend Dashboard**: Streamlit, Plotly, Custom HTML/CSS (Glassmorphism design, Montserrat/Inter typography).
- **Backend API**: Python, FastAPI, Uvicorn.
- **Predictive Engine**: Scikit-Learn (`RandomForestRegressor`), Pandas, NumPy.
- **Generative AI**: Google Gemini API (`gemini-1.5-flash` via `google-generativeai`).

---

## Getting Started

Follow these steps to run the working prototype locally on Windows:

### Step 1: Install Python 3.12
Open your terminal (PowerShell or Command Prompt) and install Python:
```powershell
winget install Python.Python.3.12
```
*Note: Restart your terminal after installation completes to refresh environment variables.*

### Step 2: Install Libraries
Navigate to the root directory `AutoChainAI_MVP` and install the python dependencies:
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Application (Dual Terminal setup)

1. **Terminal 1 (Start the Backend FastAPI Engine)**:
   ```powershell
   python -m uvicorn backend.main:app --reload
   ```
2. **Terminal 2 (Start the Streamlit Dashboard)**:
   ```powershell
   python -m streamlit run frontend/app.py
   ```

*The application will automatically launch in your browser at `http://localhost:8501`.*

<img width="1915" height="965" alt="image" src="https://github.com/user-attachments/assets/5272919d-bd18-49f4-8a21-203141bf3c57" />


---

## Demo Safety (Offline Resiliency)
This prototype has been designed for presentation reliability. If the FastAPI backend is not running or goes offline during a live demo, the Streamlit frontend **automatically switches to fallback mode**. It runs identical local simulation algorithms, ensuring all maps, gauge charts, and ML forecast lines render flawlessly without crashing.
