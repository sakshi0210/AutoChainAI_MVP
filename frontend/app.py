import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import re

# Set page config
st.set_page_config(
    page_title="AutoChain AI Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Glassmorphism & Neon Accent Design)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Montserrat:wght@400;500;600;700;800&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700 !important;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(17, 24, 39, 0.85);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background-color: rgba(31, 41, 55, 0.45);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    /* Header decoration */
    .glowing-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 0 0 30px rgba(6, 182, 212, 0.2);
    }
    
    /* Alert cards styling */
    .alert-card {
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        border-left: 4px solid;
    }
    .alert-critical {
        background-color: rgba(244, 63, 94, 0.12);
        color: #fda4af;
        border-color: #f43f5e;
        border-right: 1px solid rgba(244, 63, 94, 0.15);
        border-top: 1px solid rgba(244, 63, 94, 0.15);
        border-bottom: 1px solid rgba(244, 63, 94, 0.15);
    }
    .alert-warning {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fde047;
        border-color: #f59e0b;
        border-right: 1px solid rgba(245, 158, 11, 0.15);
        border-top: 1px solid rgba(245, 158, 11, 0.15);
        border-bottom: 1px solid rgba(245, 158, 11, 0.15);
    }
    .alert-stable {
        background-color: rgba(16, 185, 129, 0.12);
        color: #a7f3d0;
        border-color: #10b981;
        border-right: 1px solid rgba(16, 185, 129, 0.15);
        border-top: 1px solid rgba(16, 185, 129, 0.15);
        border-bottom: 1px solid rgba(16, 185, 129, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Local Simulator Fallback Database
# (If backend API is offline)
# ---------------------------------------------------------
BACKEND_URL = "http://127.0.0.1:8000"

def get_api_data(endpoint: str, method: str = "GET", payload: dict = None) -> tuple:
    """Queries backend; falls back to local simulation on connection error."""
    try:
        if method == "GET":
            r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=2.0)
        else:
            r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=2.0)
            
        if r.status_code == 200:
            return r.json(), True
    except requests.exceptions.RequestException:
        pass
    return None, False

# ---------------------------------------------------------
# Simulated Local Heuristics (Fallback Logic)
# ---------------------------------------------------------
def local_risk_score(name: str, location: str, delay: int, cost: float, geo: int, weather: int) -> dict:
    delay_score = min(100, delay * 3.5)
    cost_score = min(100, cost * 2.0)
    geo_score = geo * 10.0
    weather_score = weather * 10.0
    raw_score = (delay_score * 0.3) + (cost_score * 0.2) + (geo_score * 0.25) + (weather_score * 0.25)
    risk_score = round(min(100.0, max(0.0, raw_score)), 1)
    
    reasons = []
    if delay > 10:
        reasons.append(f"Chronic logistics delay: average delay of {delay} days detected.")
    if cost > 15:
        reasons.append(f"Severe cost creep: prices increased by {cost}% over the last quarter.")
    if geo >= 7:
        reasons.append(f"High-risk origin country ({location}): trade tariffs and border restrictions apply.")
    if weather >= 7:
        reasons.append("Active weather alerts: high vulnerability to monsoons or winter freezes.")
    if not reasons:
        reasons.append("Supplier performs within normal parameters. Low operational risks.")
        
    return {
        "supplier_name": name,
        "location": location,
        "risk_score": risk_score,
        "status": "CRITICAL RISK" if risk_score >= 70 else "MODERATE RISK" if risk_score >= 40 else "STABLE",
        "reasons": reasons,
        "metrics_breakdown": {
            "logistics_risk": round(delay_score, 1),
            "pricing_pressure": round(cost_score, 1),
            "geopolitical_vulnerability": round(geo_score, 1),
            "climate_vulnerability": round(weather_score, 1)
        }
    }

def local_alternates(location: str, score: float, material: str) -> dict:
    alternatives = [
        {"alternative_supplier": "MexiVolt Logistics Co.", "location": "Mexico", "new_risk_score": 28.0, "risk_reduction_pct": round(score - 28, 1), "cost_differential_pct": 6.5, "transit_lead_time_days": 4, "pros": ["Nearshore shipping", "USMCA tariff exemptions"], "cons": ["Slightly higher initial tooling cost"]},
        {"alternative_supplier": "Pacific Rim Battery Corp.", "location": "Vietnam", "new_risk_score": 38.0, "risk_reduction_pct": round(score - 38, 1), "cost_differential_pct": -2.0, "transit_lead_time_days": 18, "pros": ["Lower unit cost", "Highly skilled battery cluster"], "cons": ["Longer transit times"]},
        {"alternative_supplier": "Boreal Resource Partners", "location": "Canada", "new_risk_score": 15.0, "risk_reduction_pct": round(score - 15, 1), "cost_differential_pct": 11.0, "transit_lead_time_days": 3, "pros": ["Excellent political stability", "Abundant local Lithium/Nickel"], "cons": ["Premium labor rates"]}
    ]
    recommended = [alt for alt in alternatives if alt["risk_reduction_pct"] > 0]
    return {"current_location": location, "current_risk_score": score, "material": material, "alternatives": recommended}

def local_inventory_forecast(initial_stock: int, safety_stock: int, daily_supply: int, lead_time_days: int) -> dict:
    dates_hist = [(datetime.now() - timedelta(days=30-i)).strftime("%Y-%m-%d") for i in range(30)]
    hist_stock = [initial_stock + int(i * 10 - np.sin(i)*120) for i in range(30)]
    
    dates_fore = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 31)]
    predicted_demands = [100 + int(np.sin(i/3)*25 + (i * 1.5)) for i in range(1, 31)]
    
    stock_levels = []
    curr = initial_stock
    violation_idx = None
    stockout_idx = None
    
    for idx, dem in enumerate(predicted_demands):
        curr = curr + daily_supply - dem
        stock_levels.append(round(curr, 1))
        if curr < safety_stock and violation_idx is None:
            violation_idx = idx
        if curr <= 0 and stockout_idx is None:
            stockout_idx = idx
            
    reorder_date = None
    target_idx = violation_idx if violation_idx is not None else stockout_idx
    if target_idx is not None:
        target_date = datetime.now() + timedelta(days=target_idx + 1)
        reorder_date = (target_date - timedelta(days=lead_time_days)).strftime("%Y-%m-%d")
        if (target_date - timedelta(days=lead_time_days)).date() <= datetime.now().date():
            reorder_date = "IMMEDIATE REORDER (Action Required)"
            
    return {
        "historical_dates": dates_hist,
        "historical_stock": hist_stock,
        "forecast_dates": dates_fore,
        "forecast_demands": predicted_demands,
        "forecast_stock": stock_levels,
        "safety_stock": safety_stock,
        "stockout_detected": stockout_idx is not None,
        "stockout_date": dates_fore[stockout_idx] if stockout_idx is not None else None,
        "safety_violation_detected": violation_idx is not None,
        "safety_violation_date": dates_fore[violation_idx] if violation_idx is not None else None,
        "reorder_suggested_date": reorder_date,
        "shortage_units": max(0.0, float(safety_stock - min(stock_levels)))
    }

def local_commodities() -> dict:
    current_date = datetime.now()
    dates_hist = [(current_date - timedelta(days=90-i)).strftime("%Y-%m-%d") for i in range(90)]
    dates_fore = [(current_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 31)]
    
    lithium_hist = [18500 - (i*10) + np.sin(i)*250 for i in range(90)]
    lithium_fore = [lithium_hist[-1] + (i*120) for i in range(1, 31)]
    lithium_upper = [p + (200 * np.sqrt(i)) for i, p in enumerate(lithium_fore, 1)]
    lithium_lower = [max(10.0, p - (200 * np.sqrt(i))) for i, p in enumerate(lithium_fore, 1)]
    
    nickel_hist = [16200 + (i*5) + np.cos(i)*180 for i in range(90)]
    nickel_fore = [nickel_hist[-1] - (i*20) for i in range(1, 31)]
    nickel_upper = [p + (150 * np.sqrt(i)) for i, p in enumerate(nickel_fore, 1)]
    nickel_lower = [max(10.0, p - (150 * np.sqrt(i))) for i, p in enumerate(nickel_fore, 1)]
    
    copper_hist = [8800 + np.sin(i/5)*300 for i in range(90)]
    copper_fore = [copper_hist[-1] + (i*8) for i in range(1, 31)]
    copper_upper = [p + (80 * np.sqrt(i)) for i, p in enumerate(copper_fore, 1)]
    copper_lower = [max(10.0, p - (80 * np.sqrt(i))) for i, p in enumerate(copper_fore, 1)]
    
    steel_hist = [760 + np.cos(i/10)*40 for i in range(90)]
    steel_fore = [steel_hist[-1] + (i*1.5) for i in range(1, 31)]
    steel_upper = [p + (10 * np.sqrt(i)) for i, p in enumerate(steel_fore, 1)]
    steel_lower = [max(10.0, p - (10 * np.sqrt(i))) for i, p in enumerate(steel_fore, 1)]
    
    return {
        "Lithium": {"historical_dates": dates_hist, "historical_prices": lithium_hist, "forecast_dates": dates_fore, "forecast_prices": lithium_fore, "upper_forecast_bounds": lithium_upper, "lower_forecast_bounds": lithium_lower, "current_price": round(lithium_hist[-1], 2), "price_pct_change_30d": 8.4, "status": "HIGH RISK (PRICE SURGE)", "alert_reason": "Salinas environmental mining restrictions constricting supply."},
        "Nickel": {"historical_dates": dates_hist, "historical_prices": nickel_hist, "forecast_dates": dates_fore, "forecast_prices": nickel_fore, "upper_forecast_bounds": nickel_upper, "lower_forecast_bounds": nickel_lower, "current_price": round(nickel_hist[-1], 2), "price_pct_change_30d": -2.1, "status": "MODERATE RISK (SURPLUS)", "alert_reason": "Increased Indonesian smelter output saturating market."},
        "Copper": {"historical_dates": dates_hist, "historical_prices": copper_hist, "forecast_dates": dates_fore, "forecast_prices": copper_fore, "upper_forecast_bounds": copper_upper, "lower_forecast_bounds": copper_lower, "current_price": round(copper_hist[-1], 2), "price_pct_change_30d": 1.2, "status": "STABLE", "alert_reason": "Stable demand projections."},
        "Steel": {"historical_dates": dates_hist, "historical_prices": steel_hist, "forecast_dates": dates_fore, "forecast_prices": steel_fore, "upper_forecast_bounds": steel_upper, "lower_forecast_bounds": steel_lower, "current_price": round(steel_hist[-1], 2), "price_pct_change_30d": 0.5, "status": "STABLE", "alert_reason": "Automotive demand aligns with standard quotas."}
    }

def local_analyze_meeting(transcript: str) -> dict:
    action_items = [
        {"description": "Establish immediate communications with MexiVolt Logistics in Mexico regarding nearshore shipping terms.", "assignee": "Sarah (Procurement)", "deadline": "15 June", "priority": "High"},
        {"description": "Increase battery cell buffer inventory target from 1200 to 1800 units to shield against port congestion.", "assignee": "Mike (Inventory AI)", "deadline": "20 June", "priority": "High"},
        {"description": "Draft new rail cargo routes from Canada to bypass East Coast shipping delays.", "assignee": "Dave (Logistics Head)", "deadline": "30 June", "priority": "Medium"}
    ]
    risks = [
        "East Asia transit times increased by 14 days due to port bottlenecking.",
        "Expected +12% cost increase in core Lithium-Ion cells.",
        "Geopolitical tariff risks affecting semiconductor component suppliers."
    ]
    summary = "The supply alignment meeting evaluated delays and shipping costs in East Asia. The team decided to mitigate risk by sourcing up to 40% of components from Mexico and Canada, and raising safety stock thresholds for Lithium."
    return {"summary": summary, "key_risks_identified": risks, "action_items": action_items}


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-top: 10px; margin-bottom: 25px;">
        <h1 style="color: #06b6d4; margin: 0; font-size: 2.2rem; text-shadow: 0 0 15px rgba(6, 182, 212, 0.4);">AutoChain AI</h1>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">Agentic Supply Chain Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Test Backend Connection
health_data, online = get_api_data("/")
if online:
    st.sidebar.success("⚡ Engine Backend: Online")
else:
    st.sidebar.warning("⚠️ Engine Backend: Offline (Simulated Fallback Mode)")
    st.sidebar.info("Tip: Run `uvicorn backend.main:app` to start the backend.")

# Settings Section
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 LLM Credentials")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enables agentic transcript analysis via Gemini")
if gemini_key:
    st.sidebar.success("Gemini API Key Configured")
else:
    st.sidebar.info("Running standard extraction heuristic without LLM key.")

# Core navigation
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Operations Navigation")
tab_selection = st.sidebar.radio(
    "Choose Service Panel:",
    ("🎛️ Executive Dashboard", "⚠️ Risk & Sourcing Engine", "📈 Inventory & Commodities", "📝 Meeting Action Extractor")
)

# ---------------------------------------------------------
# Page Content Routing
# ---------------------------------------------------------

# --- TAB 1: Executive Dashboard ---
if tab_selection == "🎛️ Executive Dashboard":
    st.markdown('<div class="glowing-header">Operational Supply Command</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-bottom: 30px;'>Real-time AI monitoring of risk, inventory fluctuations, and critical supplier logistics.</p>", unsafe_allow_html=True)
    
    # Renders KPI Metrics
    cols = st.columns(4)
    
    # Card 1: Risk Status
    with cols[0]:
        st.markdown(
            """
            <div style="background-color: rgba(31, 41, 55, 0.45); padding: 20px; border-radius: 12px; border-left: 5px solid #f43f5e; border-top: 1px solid rgba(255, 255, 255, 0.08); border-right: 1px solid rgba(255, 255, 255, 0.08); border-bottom: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Average Supply Risk</p>
                <h2 style="margin: 5px 0 0 0; font-size: 2.2rem; color: #f8fafc; font-weight: 700;">68.4 / 100</h2>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #fda4af; font-weight: 500;">🚨 Critical (East Asia)</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    # Card 2: Active Alerts
    with cols[1]:
        st.markdown(
            """
            <div style="background-color: rgba(31, 41, 55, 0.45); padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b; border-top: 1px solid rgba(255, 255, 255, 0.08); border-right: 1px solid rgba(255, 255, 255, 0.08); border-bottom: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Active Disruptions</p>
                <h2 style="margin: 5px 0 0 0; font-size: 2.2rem; color: #f8fafc; font-weight: 700;">3 Alerts</h2>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #fde047; font-weight: 500;">⚠️ Port Congestion & Chile Mine strike</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    # Card 3: Inventory Forecast Alert
    with cols[2]:
        st.markdown(
            """
            <div style="background-color: rgba(31, 41, 55, 0.45); padding: 20px; border-radius: 12px; border-left: 5px solid #06b6d4; border-top: 1px solid rgba(255, 255, 255, 0.08); border-right: 1px solid rgba(255, 255, 255, 0.08); border-bottom: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Shortage Forecast</p>
                <h2 style="margin: 5px 0 0 0; font-size: 2.2rem; color: #f8fafc; font-weight: 700;">14 Days</h2>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #a5f3fc; font-weight: 500;">📉 Lithium Stockout Risk</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    # Card 4: Commodity Index
    with cols[3]:
        st.markdown(
            """
            <div style="background-color: rgba(31, 41, 55, 0.45); padding: 20px; border-radius: 12px; border-left: 5px solid #10b981; border-top: 1px solid rgba(255, 255, 255, 0.08); border-right: 1px solid rgba(255, 255, 255, 0.08); border-bottom: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                <p style="margin: 0; font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Commodity Volatility</p>
                <h2 style="margin: 5px 0 0 0; font-size: 2.2rem; color: #f8fafc; font-weight: 700;">+8.4%</h2>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: #a7f3d0; font-weight: 500;">📈 Lithium Price Spikes</p>
            </div>
            """, unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_graphs = st.columns([2, 1])
    
    with col_graphs[0]:
        st.markdown('<div class="glass-card"><h4>📍 Global Sourcing Distribution & Regional Risk</h4>', unsafe_allow_html=True)
        # Create a beautiful Plotly Map showing risks
        map_df = pd.DataFrame({
            "Supplier": ["Anhui Battery Co.", "Shenzhen Tech Ltd.", "MexiVolt Logistics", "Boreal Resources", "Munich Castings"],
            "Location": ["East China", "South China", "Mexico", "Canada", "Germany"],
            "Risk Score": [82, 74, 28, 15, 34],
            "Volume (units/mo)": [10000, 8000, 6000, 4000, 5000],
            "Lat": [31.86, 22.54, 25.68, 56.13, 48.13],
            "Lon": [117.28, 114.05, -100.31, -106.34, 11.58]
        })
        
        fig_map = px.scatter_mapbox(
            map_df,
            lat="Lat",
            lon="Lon",
            color="Risk Score",
            size="Volume (units/mo)",
            hover_name="Supplier",
            hover_data=["Location", "Risk Score"],
            color_continuous_scale=px.colors.sequential.Sunsetdark,
            size_max=30,
            zoom=1,
            height=380
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_graphs[1]:
        st.markdown('<div class="glass-card"><h4>⚠️ System Operations Log</h4>', unsafe_allow_html=True)
        st.markdown("""
            <div class="alert-card alert-critical">
                <strong>[CRITICAL]</strong> <strong>Lithium price spike projected.</strong> Salt-flat mining permits delayed in South America. Alternate sourcing recommended.
            </div>
            <div class="alert-card alert-warning">
                <strong>[WARNING]</strong> <strong>Port congestion alert.</strong> East Asian shipping transit delays extended by 4 days. Stockout window narrowed to 14 days.
            </div>
            <div class="alert-card alert-stable">
                <strong>[STABLE]</strong> Steel procurement lines from Canada verified. 100% capacity maintained, cost index remains flat.
            </div>
            <div class="alert-card alert-warning">
                <strong>[WARNING]</strong> Geopolitical index for China-bound cargo raised. Tariff impact +5.0% expected on raw graphite anodes.
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: Risk & Sourcing Engine ---
elif tab_selection == "⚠️ Risk & Sourcing Engine":
    st.markdown('<div class="glowing-header">Supplier Risk & Alternate Sourcing</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-bottom: 30px;'>Simulate supplier disruptions, compute composite risk indicators, and explore optimal alternative sourcing paths.</p>", unsafe_allow_html=True)
    
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        st.markdown('<div class="glass-card"><h4>⚙️ Sourcing Simulator</h4>', unsafe_allow_html=True)
        supplier_name = st.text_input("Supplier Name", "EastAsia Cathode Ltd.")
        location = st.text_input("Supplier Location", "China")
        material = st.selectbox("Critical Material", ["Lithium-Ion Battery Cell", "Nickel Alloys", "Copper Harnesses", "Structural Steel"])
        
        delay_days = st.slider("Logistics Delay (days)", 0, 30, 8, help="Average delivery delay compared to historical baseline")
        cost_inc = st.slider("Raw Material Price Increase %", 0.0, 50.0, 12.0, step=0.5)
        geo_risk = st.slider("Geopolitical Disruption Risk (0-10)", 0, 10, 7)
        weather_risk = st.slider("Severe Weather Risk (0-10)", 0, 10, 5)
        
        calculate_btn = st.button("Calculate Supplier Risk Profile", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_results:
        # If button clicked, retrieve metrics
        if calculate_btn or 'calculated_risk' not in st.session_state:
            payload = {
                "supplier_name": supplier_name,
                "location": location,
                "delivery_delay_days": delay_days,
                "cost_increase_pct": cost_inc,
                "geopolitical_risk": geo_risk,
                "weather_disruption_level": weather_risk
            }
            
            if online:
                res, success = get_api_data("/api/risk/score", method="POST", payload=payload)
                if not success:
                    res = local_risk_score(supplier_name, location, delay_days, cost_inc, geo_risk, weather_risk)
            else:
                res = local_risk_score(supplier_name, location, delay_days, cost_inc, geo_risk, weather_risk)
                
            st.session_state['calculated_risk'] = res
            
        res = st.session_state['calculated_risk']
        
        # Display Results
        risk_score = res["risk_score"]
        status = res["status"]
        reasons = res["reasons"]
        breakdown = res["metrics_breakdown"]
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader(f"🛡️ {res['supplier_name']} Profile")
            st.write(f"**Origin:** {res['location']} | **Material:** {material}")
            
            # Risk level color badge
            badge_color = "#f43f5e" if risk_score >= 70 else "#f59e0b" if risk_score >= 40 else "#10b981"
            st.markdown(
                f"<span style='background-color:{badge_color}22; color:{badge_color}; border: 1px solid {badge_color}; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.9rem;'>{status}</span>",
                unsafe_allow_html=True
            )
            
            st.markdown("<br><br>**Primary Risk Drivers:**", unsafe_allow_html=True)
            for r in reasons:
                st.markdown(f"❌ {r}")
                
        with c2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Composite Risk Index", 'font': {'size': 16, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': badge_color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(244, 63, 94, 0.15)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#f8fafc', 'family': 'Inter'},
                height=220,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- ALTERNATIVE SOURCING SUGGESTIONS ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💡 Optimal Alternative Sourcing Paths")
        
        alt_payload = {
            "current_location": res["location"],
            "current_risk_score": risk_score,
            "material": material
        }
        
        if online:
            alt_res, success = get_api_data("/api/risk/alternates", method="POST", payload=alt_payload)
            if not success:
                alt_res = local_alternates(res["location"], risk_score, material)
        else:
            alt_res = local_alternates(res["location"], risk_score, material)
            
        alternatives = alt_res["alternatives"]
        
        if not alternatives:
            st.info("Current risk profile is optimal. No higher-efficiency alternatives needed.")
        else:
            # Render comparison chart
            alt_names = [a["alternative_supplier"] for a in alternatives] + [res["supplier_name"]]
            alt_risks = [a["new_risk_score"] for a in alternatives] + [risk_score]
            alt_colors = ["#10b981"] * len(alternatives) + [badge_color]
            
            fig_comp = go.Figure(data=[
                go.Bar(
                    x=alt_risks,
                    y=alt_names,
                    orientation='h',
                    marker_color=alt_colors,
                    text=alt_risks,
                    textposition='auto',
                )
            ])
            fig_comp.update_layout(
                title="Risk Level Comparison (Lower is Better)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#f8fafc'},
                xaxis_title="Risk Score",
                yaxis_title="Supplier",
                height=180 + len(alternatives)*30,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Render alternate cards
            st.markdown("<h5 style='margin-top:20px;'>Alternative Sourcing Trade-Off Matrix:</h5>", unsafe_allow_html=True)
            
            for alt in alternatives:
                st.markdown(f"""
                    <div style="background-color: rgba(17, 24, 39, 0.4); padding: 15px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <span style="font-weight: 700; font-size: 1.05rem; color:#06b6d4;">{alt['alternative_supplier']} ({alt['location']})</span>
                            <div>
                                <span style="background-color: rgba(16, 185, 129, 0.15); color: #10b981; padding: 3px 8px; border-radius: 12px; font-size:0.75rem; font-weight:700; margin-right: 5px;">Risk Reduction: -{alt['risk_reduction_pct']}%</span>
                                <span style="background-color: {'rgba(244, 63, 94, 0.15); color: #f43f5e;' if alt['cost_differential_pct'] > 0 else 'rgba(16, 185, 129, 0.15); color: #10b981;'} padding: 3px 8px; border-radius: 12px; font-size:0.75rem; font-weight:700;">Cost: {'+' if alt['cost_differential_pct'] > 0 else ''}{alt['cost_differential_pct']}%</span>
                            </div>
                        </div>
                        <p style="margin: 8px 0; font-size:0.85rem; color:#94a3b8;"><strong>Lead Time:</strong> {alt['transit_lead_time_days']} days</p>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                            <div><span style="color:#10b981; font-weight:600; font-size:0.85rem;">✅ Pros:</span> <span style="font-size:0.8rem; color:#cbd5e1;">{', '.join(alt['pros'])}</span></div>
                            <div><span style="color:#f43f5e; font-weight:600; font-size:0.85rem;">❌ Cons:</span> <span style="font-size:0.8rem; color:#cbd5e1;">{', '.join(alt['cons'])}</span></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: Inventory & Commodities ---
elif tab_selection == "📈 Inventory & Commodities":
    st.markdown('<div class="glowing-header">Predictive Planning & Resource Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-bottom: 30px;'>Apply Machine Learning algorithms to predict demand volatility and map future resource price levels.</p>", unsafe_allow_html=True)
    
    col_inputs, col_forecasting = st.columns([1, 2])
    
    with col_inputs:
        st.markdown('<div class="glass-card"><h4>📉 Inventory Model Config</h4>', unsafe_allow_html=True)
        initial_stock = st.number_input("Current Warehouse Stock", value=1200, step=50)
        safety_stock = st.number_input("Safety Stock Threshold", value=400, step=50)
        daily_supply = st.number_input("Inbound Supplier Lead Delivery Rate (units/day)", value=100, step=5)
        lead_time = st.slider("Supplier Lead Time (days)", 2, 20, 8)
        
        forecast_btn = st.button("Generate ML Forecast", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_forecasting:
        # Load forecast
        if forecast_btn or 'calculated_forecast' not in st.session_state:
            if online:
                res_f, success = get_api_data(f"/api/inventory/forecast?initial_stock={initial_stock}&safety_stock={safety_stock}&daily_supply={daily_supply}&lead_time_days={lead_time}")
                if not success:
                    res_f = local_inventory_forecast(initial_stock, safety_stock, daily_supply, lead_time)
            else:
                res_f = local_inventory_forecast(initial_stock, safety_stock, daily_supply, lead_time)
                
            st.session_state['calculated_forecast'] = res_f
            
        res_f = st.session_state['calculated_forecast']
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📦 Inventory Levels & Shortage Forecast")
        
        # Display alerts
        if res_f["stockout_detected"]:
            st.markdown(
                f"""<div class="alert-card alert-critical">
                    <strong>⚠️ CRITICAL CRITICAL SHORTAGE DETECTED:</strong> Stockout projected to hit on <strong>{res_f['stockout_date']}</strong>. Shortage reaches <strong>{res_f['shortage_units']} units</strong> below Safety Stock limits.
                </div>""", unsafe_allow_html=True
            )
        elif res_f["safety_violation_detected"]:
            st.markdown(
                f"""<div class="alert-card alert-warning">
                    <strong>⚠️ SAFETY STOCK ALERT:</strong> Inbound supply will fall below Safety Stock limits on <strong>{res_f['safety_violation_date']}</strong>.
                </div>""", unsafe_allow_html=True
            )
        else:
            st.markdown(
                """<div class="alert-card alert-stable">
                    <strong>✅ STOCK STABILITY CONFIRMED:</strong> Projected demand does not challenge warehouse safety thresholds over the next 30 days.
                </div>""", unsafe_allow_html=True
            )
            
        # Display reorder date
        reorder_date = res_f["reorder_suggested_date"]
        if reorder_date:
            st.markdown(f"**⏰ Optimal Reorder Trigger Date:** <span style='font-size: 1.15rem; color:#f59e0b; font-weight:700;'>{reorder_date}</span>", unsafe_allow_html=True)
            
        # Build Plotly Plot for Inventory levels
        fig_inv = go.Figure()
        
        # Historical stock
        fig_inv.add_trace(go.Scatter(
            x=res_f["historical_dates"],
            y=res_f["historical_stock"],
            name="Historical Stock Level",
            line=dict(color="#3b82f6", width=2.5)
        ))
        
        # Forecasted stock
        fig_inv.add_trace(go.Scatter(
            x=res_f["forecast_dates"],
            y=res_f["forecast_stock"],
            name="Forecasted Stock Level (ML Prediction)",
            line=dict(color="#06b6d4", width=3, dash='dash')
        ))
        
        # Safety Stock Line
        fig_inv.add_trace(go.Scatter(
            x=res_f["historical_dates"] + res_f["forecast_dates"],
            y=[res_f["safety_stock"]] * len(res_f["historical_dates"] + res_f["forecast_dates"]),
            name="Safety Stock Line",
            line=dict(color="#ef4444", width=1.5, dash='dot')
        ))
        
        fig_inv.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f8fafc'},
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Date",
            yaxis_title="Stock Units",
            height=300,
            hovermode="x unified"
        )
        fig_inv.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
        fig_inv.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
        
        st.plotly_chart(fig_inv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- COMMODITY INTELLIGENCE SECTION ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💎 Commodity Price Forecasts & Supply Indicators")
    
    if online:
        comm_data, success = get_api_data("/api/commodities")
        if not success:
            comm_data = local_commodities()
    else:
        comm_data = local_commodities()
        
    col_comm_sel, col_comm_chart = st.columns([1, 2])
    
    with col_comm_sel:
        comm_choice = st.radio("Select Resource Asset:", list(comm_data.keys()))
        data_c = comm_data[comm_choice]
        
        # Display commodity KPIs
        st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
        st.write(f"**Spot Price:** ${data_c['current_price']:,} / ton")
        
        vol_pct = data_c['price_pct_change_30d']
        vol_color = "#f43f5e" if vol_pct > 5.0 else "#10b981" if vol_pct < 0 else "#f59e0b"
        st.markdown(f"**30d Trailing Trend:** <span style='color:{vol_color}; font-weight:700;'>{'+' if vol_pct > 0 else ''}{vol_pct}%</span>", unsafe_allow_html=True)
        
        status_color = "#f43f5e" if "HIGH" in data_c['status'] else "#f59e0b" if "MODERATE" in data_c['status'] else "#10b981"
        st.markdown(
            f"<div style='border: 1px solid {status_color}; background-color:{status_color}11; color:{status_color}; font-weight:700; text-align:center; padding: 5px; border-radius: 8px; font-size:0.8rem; margin:10px 0;'>{data_c['status']}</div>",
            unsafe_allow_html=True
        )
        st.write(f"ℹ️ **Price Driver:** {data_c['alert_reason']}")
        
    with col_comm_chart:
        # Plot commodity forecasting
        fig_comm = go.Figure()
        
        # Historical
        fig_comm.add_trace(go.Scatter(
            x=data_c["historical_dates"],
            y=data_c["historical_prices"],
            name="Historical Spot Price",
            line=dict(color="#8b5cf6", width=2.5)
        ))
        
        # Forecast
        fig_comm.add_trace(go.Scatter(
            x=data_c["forecast_dates"],
            y=data_c["forecast_prices"],
            name="Forecasted Trend",
            line=dict(color="#d946ef", width=3, dash='dash')
        ))
        
        # Confidence Band Shading
        fig_comm.add_trace(go.Scatter(
            x=data_c["forecast_dates"] + data_c["forecast_dates"][::-1],
            y=data_c["upper_forecast_bounds"] + data_c["lower_forecast_bounds"][::-1],
            fill='toself',
            fillcolor='rgba(217, 70, 239, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Forecast Uncertainty Interval"
        ))
        
        fig_comm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f8fafc'},
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="Date",
            yaxis_title="USD ($) per Ton",
            height=260,
            hovermode="x unified"
        )
        fig_comm.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
        fig_comm.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
        
        st.plotly_chart(fig_comm, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: AI Meeting Summarizer ---
elif tab_selection == "📝 Meeting Action Extractor":
    st.markdown('<div class="glowing-header">AI Meeting Action Extractor</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-bottom: 30px;'>Extract procurement actions, deadlines, owners, and risk assessments automatically from meeting transcripts using generative AI models.</p>", unsafe_allow_html=True)
    
    col_input_text, col_actions_display = st.columns([1, 1])
    
    with col_input_text:
        st.markdown('<div class="glass-card"><h4>🗣️ Meeting Transcript Input</h4>', unsafe_allow_html=True)
        
        # Sample transcript for fast demo
        default_transcript = """AutoChain Procurement Sync - 4 June 2026

John (Operations): The logistics backlog at Shenzhen port is extending. Deliveries from Shenzhen Tech Ltd are taking 12 days longer than normal, causing severe warehouse stockout risks for Lithium-Ion battery cell. We need an alternative ASAP.

Sarah (Procurement): I have initialized talks with MexiVolt Logistics in Mexico. Sourcing from Mexico cuts transportation transit lead time to 4 days, but would incur a 6.5% tool and die setup cost differential. We need to reach out to them to verify unit pricing. Sarah will contact them by 15 June.

Mike (Inventory): If Shenzhen delays continue, we will breach our safety stock of 400 battery packs in 14 days. I strongly recommend Mike increases the safety targets to 1800 units for June to create a buffer. We need to get warehouse clearance for this capacity by next Friday.

Dave (Logistics): We should also check Rail logistics corridors. Dave will draft cargo routes from Boreal Resources in Canada by 30 June as a secondary buffer.
"""
        
        transcript_text = st.text_area("Paste procurement meeting logs here:", default_transcript, height=300)
        
        trigger_extract = st.button("Extract Actions & Risks", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_actions_display:
        if trigger_extract or 'extracted_meeting' not in st.session_state:
            payload = {
                "transcript": transcript_text,
                "gemini_api_key": gemini_key if gemini_key else None
            }
            
            if online:
                res_m, success = get_api_data("/api/analyze-meeting", method="POST", payload=payload)
                if not success:
                    res_m = local_analyze_meeting(transcript_text)
            else:
                res_m = local_analyze_meeting(transcript_text)
                
            st.session_state['extracted_meeting'] = res_m
            
        res_m = st.session_state['extracted_meeting']
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📝 Executive Summary")
        st.markdown(f"<p style='color:#cbd5e1; font-size:0.95rem; line-height: 1.5;'>{res_m['summary']}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_sub_1, col_sub_2 = st.columns(2)
        
        with col_sub_1:
            st.markdown('<div class="glass-card" style="padding:15px; min-height: 250px;">', unsafe_allow_html=True)
            st.markdown("<h5 style='color: #fda4af; margin-bottom:12px;'>⚠️ Identified Risks</h5>", unsafe_allow_html=True)
            for r in res_m["key_risks_identified"]:
                st.markdown(f"• <span style='font-size:0.85rem; color:#e2e8f0;'>{r}</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_sub_2:
            st.markdown('<div class="glass-card" style="padding:15px; min-height: 250px;">', unsafe_allow_html=True)
            st.markdown("<h5 style='color: #06b6d4; margin-bottom:12px;'>⚡ Action Items Checklist</h5>", unsafe_allow_html=True)
            
            for idx, action in enumerate(res_m["action_items"]):
                # Render priority color indicator
                pri = action.get("priority", "Medium")
                pri_color = "#f43f5e" if pri == "High" else "#f59e0b" if pri == "Medium" else "#10b981"
                
                # Streamlit checkbox to interact
                checked = st.checkbox(
                    f"{action['description']}", 
                    key=f"task_{idx}",
                    help=f"Owner: {action['assignee']} | Deadline: {action['deadline']}"
                )
                
                st.markdown(
                    f"""<div style="margin-left: 25px; margin-top:-5px; margin-bottom: 12px; font-size: 0.75rem; color:#94a3b8;">
                        👤 <strong>Assignee:</strong> {action['assignee']} | 📅 <strong>Due:</strong> {action['deadline']} | 
                        <span style="color: {pri_color}; font-weight:700;">▲ {pri}</span>
                    </div>""", unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)