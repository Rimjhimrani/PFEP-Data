import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io

# --- CUSTOM CSS FOR MODERN UI ---
def load_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom title styling */
    .custom-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Step container styling */
    .step-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .step-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Process step styling */
    .process-step {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #e3f2fd;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .process-step:hover {
        border-color: #2196f3;
        transform: translateY(-1px);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-pending {
        background-color: #ff9800;
        animation: pulse 2s infinite;
    }
    
    .status-complete {
        background-color: #4caf50;
    }
    
    .status-running {
        background-color: #2196f3;
        animation: spin 1s linear infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Primary action button */
    .primary-button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        font-size: 1.1rem !important;
        padding: 0.8rem 3rem !important;
        border-radius: 30px !important;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3) !important;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }
    
    .info-card h3 {
        margin-top: 0;
        color: white !important;
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.2);
    }
    
    /* Warning card */
    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.2);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    /* Process flow indicator */
    .process-flow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-radius: 12px;
    }
    
    .flow-step {
        text-align: center;
        flex: 1;
        padding: 0.5rem;
    }
    
    .flow-arrow {
        color: #667eea;
        font-size: 1.5rem;
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def create_step_indicator(current_step, total_steps):
    """Create a visual step indicator"""
    steps = []
    for i in range(1, total_steps + 1):
        if i < current_step:
            status = "complete"
            icon = "✅"
        elif i == current_step:
            status = "running"
            icon = "🔄"
        else:
            status = "pending"
            icon = "⏳"
        
        steps.append(f"""
        <div class="flow-step">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-weight: 600;">Step {i}</div>
        </div>
        """)
    
    arrows = ["<div class='flow-arrow'>→</div>"] * (total_steps - 1)
    
    flow_html = "<div class='process-flow'>"
    for i, step in enumerate(steps):
        flow_html += step
        if i < len(arrows):
            flow_html += arrows[i]
    flow_html += "</div>"
    
    st.markdown(flow_html, unsafe_allow_html=True)

def create_info_card(title, content, card_type="info"):
    """Create styled info cards"""
    card_class = f"{card_type}-card"
    st.markdown(f"""
    <div class="{card_class}">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_display(metrics_data):
    """Create styled metric display"""
    cols = st.columns(len(metrics_data))
    for i, (label, value, delta) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0 0 0.5rem 0;">{label}</h3>
                <h2 style="color: #333; margin: 0;">{value}</h2>
                {f'<p style="color: #4caf50; margin: 0.5rem 0 0 0;">+{delta}</p>' if delta else ''}
            </div>
            """, unsafe_allow_html=True)

# --- 1. MASTER TEMPLATE AND LOGIC CONSTANTS ---

ALL_TEMPLATE_COLUMNS = [
    'SR.NO', 'PARTNO', 'PART DESCRIPTION', 'Qty/Veh 1', 'Qty/Veh 2', 'TOTAL', 'UOM', 'ST.NO',
    'FAMILY', 'Qty/Veh 1_Daily', 'Qty/Veh 2_Daily', 'NET', 'UNIT PRICE', 'PART CLASSIFICATION',
    'L-MM_Size', 'W-MM_Size', 'H-MM_Size', 'Volume (m^3)', 'SIZE CLASSIFICATION', 'VENDOR CODE',
    'VENDOR NAME', 'VENDOR TYPE', 'CITY', 'STATE', 'COUNTRY', 'PINCODE', 'PRIMARY PACK TYPE',
    'L-MM_Prim_Pack', 'W-MM_Prim_Pack', 'H-MM_Prim_Pack', 'QTY/PACK_Prim', 'PRIM. PACK LIFESPAN',
    'PRIMARY PACKING FACTOR', 'SECONDARY PACK TYPE', 'L-MM_Sec_Pack', 'W-MM_Sec_Pack',
    'H-MM_Sec_Pack', 'NO OF BOXES', 'QTY/PACK_Sec', 'SEC. PACK LIFESPAN', 'ONE WAY/ RETURNABLE',
    'DISTANCE CODE', 'INVENTORY CLASSIFICATION', 'RM IN DAYS', 'RM IN QTY',
    'RM IN INR', 'PACKING FACTOR (PF)', 'NO OF SEC. PACK REQD.', 'NO OF SEC REQ. AS PER PF',
    'WH LOC', 'PRIMARY LOCATION ID', 'SECONDARY LOCATION ID',
    'OVER FLOW TO BE ALLOTED', 'DOCK NUMBER', 'STACKING FACTOR', 'SUPPLY TYPE', 'SUPPLY VEH SET',
    'SUPPLY STRATEGY', 'SUPPLY CONDITION', 'CONTAINER LINE SIDE', 'L-MM_Supply', 'W-MM_Supply',
    'H-MM_Supply', 'Volume_Supply', 'QTY/CONTAINER -LS -9M', 'QTY/CONTAINER -LS-12M', 'STORAGE LINE SIDE',
    'L-MM_Line', 'W-MM_Line', 'H-MM_Line', 'Volume_Line', 'CONTAINER / RACK','NO OF TRIPS/DAY', 'INVENTORY LINE SIDE'
]


# Enhanced column mapping to include qty/veh variations
PFEP_COLUMN_MAP = {
    'part_id': 'PARTNO', 'description': 'PART DESCRIPTION', 'qty_veh': 'Qty/Veh',
    'qty/veh': 'Qty/Veh', 'quantity_per_vehicle': 'Qty/Veh', 'net_daily_consumption': 'NET',
    'unit_price': 'UNIT PRICE', 'vendor_code': 'VENDOR CODE', 'vendor_name': 'VENDOR NAME',
    'city': 'CITY', 'state': 'STATE', 'country': 'COUNTRY', 'pincode': 'PINCODE',
    'length': 'L-MM_Size', 'width': 'W-MM_Size', 'height': 'H-MM_Size',
    'qty_per_pack': 'QTY/PACK_Sec', 'packing_factor': 'PACKING FACTOR (PF)',
    'primary_packaging_factor': 'PRIMARY PACKING FACTOR'
}

INTERNAL_TO_PFEP_NEW_COLS = {
    'family': 'FAMILY', 'part_classification': 'PART CLASSIFICATION', 'volume_m3': 'Volume (m^3)',
    'size_classification': 'SIZE CLASSIFICATION', 'wh_loc': 'WH LOC'
}

FAMILY_KEYWORD_MAPPING = {
    "ADAPTOR": ["ADAPTOR", "ADAPTER"], "Beading": ["BEADING"],
    "Electrical": ["BATTERY", "HVPDU", "ELECTRICAL", "INVERTER", "SENSOR", "DC", "COMPRESSOR", "TMCS", "COOLING", "BRAKE SIGNAL", "VCU", "VEHICLE CONTROL", "EVCC", "EBS ECU", "ECU", "CONTROL UNIT", "SIGNAL", "TRANSMITTER", "TRACTION", "HV", "KWH", "EBS", "SWITCH", "HORN"],
    "Electronics": ["DISPLAY", "APC", "SCREEN", "MICROPHONE", "CAMERA", "SPEAKER", "DASHBOARD", "ELECTRONICS", "SSD", "WOODWARD", "FDAS", "BDC", "GEN-2", "SENSOR", "BUZZER"],
    "Wheels": ["WHEEL", "TYRE", "TIRE", "RIM"], "Harness": ["HARNESS", "CABLE"], "Mechanical": ["PUMP", "SHAFT", "LINK", "GEAR", "ARM"],
    "Hardware": ["NUT", "BOLT", "SCREW", "WASHER", "RIVET", "M5", "M22", "M12", "CLAMP", "CLIP", "CABLE TIE", "DIN", "ZFP"],
    "Bracket": ["BRACKET", "BRKT", "BKT", "BRCKT"], "ASSY": ["ASSY"], "Sticker": ["STICKER", "LOGO", "EMBLEM"], "Suspension": ["SUSPENSION"],
    "Tank": ["TANK"], "Tape": ["TAPE", "REFLECTOR", "COLOUR"], "Tool Kit": ["TOOL KIT"], "Valve": ["VALVE"], "Hose": ["HOSE"],
    "Insulation": ["INSULATION"], "Interior & Exterior": ["ROLLER", "FIRE", "HAMMER"], "L-angle": ["L-ANGLE"], "Lamp": ["LAMP"], "Lock": ["LOCK"],
    "Lubricants": ["GREASE", "LUBRICANT"], "Medical": ["MEDICAL", "FIRST AID"], "Mirror": ["MIRROR", "ORVM"], "Motor": ["MOTOR"],
    "Mounting": ["MOUNT", "MTG", "MNTG", "MOUNTED"], "Oil": ["OIL"], "Panel": ["PANEL"], "Pillar": ["PILLAR"],
    "Pipe": ["PIPE", "TUBE", "SUCTION", "TUBULAR"], "Plate": ["PLATE"], "Plywood": ["FLOORING", "PLYWOOD", "EPGC"], "Profile": ["PROFILE", "ALUMINIUM"],
    "Rail": ["RAIL"], "Rubber": ["RUBBER", "GROMMET", "MOULDING"], "Seal": ["SEAL"], "Seat": ["SEAT"], "ABS Cover": ["ABS COVER"], "AC": ["AC"],
    "ACP Sheet": ["ACP SHEET"], "Aluminium": ["ALUMINIUM", "ALUMINUM"], "AXLE": ["AXLE"], "Bush": ["BUSH"], "Chassis": ["CHASSIS"],
    "Dome": ["DOME"], "Door": ["DOOR"], "Filter": ["FILTER"], "Flap": ["FLAP"], "FRP": ["FRP", "FACIA"], "Glass": ["GLASS", "WINDSHIELD", "WINDSHILED"],
    "Handle": ["HANDLE", "HAND", "PLASTIC"], "HATCH": ["HATCH"], "HDF Board": ["HDF"]
}
CATEGORY_PRIORITY_FAMILIES = {"ACP Sheet", "ADAPTOR", "Bracket", "Bush", "Flap", "Handle", "Beading", "Lubricants", "Panel", "Pillar", "Rail", "Seal", "Sticker", "Valve"}
BASE_WAREHOUSE_MAPPING = {
    "ABS Cover": "HRR", "ADAPTOR": "MEZ B-01(A)", "Beading": "HRR", "AXLE": "FLOOR", "Bush": "HRR", "Chassis": "FLOOR", "Dome": "MEZ C-02(B)", "Door": "MRR(C-01)",
    "Electrical": "HRR", "Filter": "CRL", "Flap": "MEZ C-02", "Insulation": "MEZ C-02(B)", "Interior & Exterior": "HRR", "L-angle": "MEZ B-01(A)", "Lamp": "CRL",
    "Lock": "CRL", "Lubricants": "HRR", "Medical": "HRR", "Mirror": "HRR", "Motor": "HRR", "Mounting": "HRR", "Oil": "HRR", "Panel": "MEZ C-02", "Pillar": "MEZ C-02",
    "Pipe": "HRR", "Plate": "HRR", "Profile": "HRR", "Rail": "CTR(C-01)", "Seal": "HRR", "Seat": "MRR(C-01)", "Sticker": "MEZ B-01(A)", "Suspension": "MRR(C-01)",
    "Tank": "HRR", "Tool Kit": "HRR", "Valve": "CRL", "Wheels": "HRR", "Hardware": "MEZ B-02(A)", "Glass": "MRR(C-01)", "Harness": "HRR", "Hose": "HRR",
    "Aluminium": "HRR", "ACP Sheet": "MEZ C-02(B)", "Handle": "HRR", "HATCH": "HRR", "HDF Board": "MRR(C-01)", "FRP": "CTR", "Others": "HRR"
}
WAREHOUSE_LOCATION_FULL_FORMS = {
    "CRL": "CAROUSAL", "HRR": "HIGH RACK", "MEZ B-01": "MEZANNAINE B-01", "MEZ B-01(A)": "MEZANNAINE B-01(A)", "MEZ B-02": "MEZANNAINE B-02",
    "MEZ B-02(A)": "MEZANNAINE B-02(A)", "MEZ C-02": "MEZANNAINE C-02", "MEZ C-02(B)": "MEZANNAINE C-02(B)", "MRR": "MID RISE RACK",
    "MRR(C-01)": "MID RISE RACK (C-01)", "CTR": "CANTILEVER RACK (FASCIA)", "CTR(C-01)": "CANTILEVER RACK (C-01)", "FLOOR": "FLOOR", "OUTSIDE": "OUTSIDE",
    "DIRECT FROM INSTOR": "DIRECT FROM INSTOR"
}


# --- DISTANCE CALCULATION COMPONENTS ---
@st.cache_data
def get_lat_lon(_geolocator, pincode, country="India", city="", state="", retries=3, backoff_factor=2):
    pincode_str = str(pincode).strip().split('.')[0]
    if not pincode_str.isdigit() or int(pincode_str) == 0:
        return (None, None)

    query = f"{pincode_str}, {country}"
    if city and state:
        query = f"{pincode_str}, {city}, {state}, {country}"

    for attempt in range(retries):
        try:
            time.sleep(1)
            location = _geolocator.geocode(query, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff_factor * (attempt + 1))
            continue
    return (None, None)

def get_distance_code(distance):
    if pd.isna(distance): return None
    if distance < 50: return 1
    if distance <= 250: return 2
    if distance <= 750: return 3
    return 4

# --- 2. DATA LOADING AND CONSOLIDATION ---
def read_uploaded_file(uploaded_file):
    """Reads an uploaded file into a pandas DataFrame."""
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            return pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        st.warning(f"Unsupported file type: {uploaded_file.name}. Please use CSV or Excel.")
        return None
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None

def find_qty_veh_column(df):
    possible_names = [
        'qty/veh', 'Qty/Veh', 'QTY/VEH', 'qty_veh', 'Qty_Veh', 'quantity/vehicle',
        'Quantity/Vehicle', 'QUANTITY/VEHICLE', 'qty per veh', 'Qty Per Veh',
        'QTY PER VEH', 'vehicle qty', 'Vehicle Qty', 'VEHICLE QTY'
    ]
    for col in df.columns:
        if col in possible_names or str(col).lower().strip() in [p.lower() for p in possible_names]:
            return col
    return None

def find_and_rename_columns(df, file_number=None):
    rename_dict, found_keys = {}, []
    qty_veh_col = find_qty_veh_column(df)
    if qty_veh_col:
        target_col = {1: 'qty_veh_1', 2: 'qty_veh_2'}.get(file_number, 'qty_veh')
        rename_dict[qty_veh_col] = target_col
        found_keys.append(target_col)

    for internal_key, pfep_name in PFEP_COLUMN_MAP.items():
        if internal_key in ['qty_veh', 'qty/veh', 'quantity_per_vehicle']: continue
        if pfep_name in df.columns:
            rename_dict[pfep_name] = internal_key
            found_keys.append(internal_key)
    df.rename(columns=rename_dict, inplace=True)
    return df, 'part_id' in found_keys

def process_and_diagnose_qty_columns(df):
    log = []
    for col_name, default_name in [('qty_veh_1', 'Qty/Veh 1'), ('qty_veh_2', 'Qty/Veh 2')]:
        if col_name not in df.columns:
            log.append(f"INFO: '{default_name}' column not found. Creating it with all zeros.")
            df[col_name] = 0
        else:
            numeric_col = pd.to_numeric(df[col_name], errors='coerce')
            invalid_count = numeric_col.isna().sum()
            if invalid_count > 0:
                log.append(f"⚠️ WARNING: Found {invalid_count} rows in '{default_name}' that were blank or non-numeric. These have been set to 0.")
            df[col_name] = numeric_col.fillna(0)
    return df, log

def consolidate_data(bom_files, vendor_file, daily_mult_1, daily_mult_2):
    all_boms, logs = [], []
    file_counter = 0

    for bom_type, files in bom_files.items():
        for file in files:
            df = read_uploaded_file(file)
            if df is None: continue
            file_counter += 1
            df, has_part_id = find_and_rename_columns(df, file_counter if bom_type == "MBOM" else None)
            if has_part_id:
                all_boms.append(df)
                logs.append(f"✅ Added {len(df)} records from '{file.name}'.")
            else:
                logs.append(f"❌ Part ID ('{PFEP_COLUMN_MAP['part_id']}') is essential. Skipping file '{file.name}'.")
                file_counter -= 1

    if not all_boms: return None, logs
    master_bom = all_boms[0].copy()
    for i, df in enumerate(all_boms[1:], 2):
        master_bom = pd.merge(master_bom, df, on='part_id', how='outer', suffixes=('', f'_dup{i}'))
        for col in [c for c in master_bom.columns if f'_dup{i}' in c]:
            original_col = col.replace(f'_dup{i}', '')
            master_bom[original_col] = master_bom[original_col].fillna(master_bom[col])
            master_bom.drop(columns=[col], inplace=True)

    master_bom, qty_logs = process_and_diagnose_qty_columns(master_bom)
    logs.extend(qty_logs)
    master_bom['total_qty'] = master_bom['qty_veh_1'] + master_bom['qty_veh_2']
    master_bom['qty_veh_1_daily'] = master_bom['qty_veh_1'] * daily_mult_1
    master_bom['qty_veh_2_daily'] = master_bom['qty_veh_2'] * daily_mult_2
    master_bom['net_daily_consumption'] = master_bom['qty_veh_1_daily'] + master_bom['qty_veh_2_daily']
    logs.append(f"✅ Qty/Veh calculations complete. Total unique parts: {len(master_bom)}")

    final_df = master_bom
    if vendor_file:
        vendor_df = read_uploaded_file(vendor_file)
        if vendor_df is not None:
            vendor_df, has_part_id = find_and_rename_columns(vendor_df)
            if has_part_id:
                vendor_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
                final_df = pd.merge(master_bom, vendor_df, on='part_id', how='left', suffixes=('', '_vendor'))
                for col in [c for c in final_df.columns if '_vendor' in c]:
                    original_col = col.replace('_vendor', '')
                    final_df[original_col] = final_df[original_col].fillna(final_df[col])
                    final_df.drop(columns=[col], inplace=True)
                logs.append("✅ Vendor data successfully merged.")
            else:
                logs.append("⚠️ Part ID not found in vendor file. Cannot merge.")
    final_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
    return final_df, logs


# --- 3. NEW PERCENTAGE-BASED PART CLASSIFICATION SYSTEM ---
class PartClassificationSystem:
    def __init__(self):
        self.percentages = {'C': {'target': 60, 'tolerance': 5}, 'B': {'target': 25, 'tolerance': 2}, 'A': {'target': 12, 'tolerance': 2}, 'AA': {'target': 3, 'tolerance': 1}}
        self.calculated_ranges = {}
    def load_data_from_dataframe(self, df, price_column='unit_price', part_id_column='part_id'):
        self.parts_data = df.copy()
        self.price_column, self.part_id_column = price_column, part_id_column
        self.calculate_percentage_ranges()
    def calculate_percentage_ranges(self):
        valid_prices = pd.to_numeric(self.parts_data[self.price_column], errors='coerce').dropna().sort_values()
        if valid_prices.empty: return
        total_valid_parts = len(valid_prices)
        ranges, cumulative_percent = {}, 0
        for class_name in ['C', 'B', 'A', 'AA']:
            target_percent = self.percentages[class_name]['target']
            start_idx = int((cumulative_percent / 100) * total_valid_parts)
            end_idx = int(((cumulative_percent + target_percent) / 100) * total_valid_parts) - 1
            start_idx, end_idx = max(0, start_idx), min(end_idx, total_valid_parts - 1)
            if start_idx <= end_idx: ranges[class_name] = {'min': valid_prices.iloc[start_idx], 'max': valid_prices.iloc[end_idx], 'count': end_idx - start_idx + 1}
            else: ranges[class_name] = {'min': None, 'max': None, 'count': 0}
            cumulative_percent += target_percent
        self.calculated_ranges = ranges
    def classify_part(self, unit_price):
        if pd.isna(unit_price): return 'Manual'
        try: unit_price = float(unit_price)
        except (ValueError, TypeError): return 'Manual'
        for class_name in ['AA', 'A', 'B', 'C']:
            range_info = self.calculated_ranges.get(class_name)
            if range_info and range_info['min'] is not None and range_info['min'] <= unit_price <= range_info['max']:
                return class_name
        return 'Unclassified'
    def classify_all_parts(self):
        if not hasattr(self, 'parts_data'): return None
        return self.parts_data[self.price_column].apply(self.classify_part)

# --- 4. DATA PROCESSING CLASS ---
class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data
        self.rm_days_mapping = {'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11, 'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16, 'C1': 16, 'C2': 31}
        self.classifier = PartClassificationSystem()
        self.geolocator = Nominatim(user_agent=f"inventory_streamlit_app_{time.time()}")
    def run_family_classification(self):
        if 'description' not in self.data.columns: return
        def find_kw_pos(desc, kw):
            match = re.search(r'\b' + re.escape(str(kw).upper()) + r'\b', str(desc).upper())
            return match.start() if match else -1
        def extract_family(desc):
            if pd.isna(desc): return 'Others'
            for fam in CATEGORY_PRIORITY_FAMILIES:
                if fam in FAMILY_KEYWORD_MAPPING and any(find_kw_pos(desc, kw) != -1 for kw in FAMILY_KEYWORD_MAPPING[fam]): return fam
            matches = [(pos, fam) for fam, kws in FAMILY_KEYWORD_MAPPING.items() if fam not in CATEGORY_PRIORITY_FAMILIES for kw in kws for pos in [find_kw_pos(desc, kw)] if pos != -1]
            return min(matches, key=lambda x: x[0])[1] if matches else 'Others'
        self.data['family'] = self.data['description'].apply(extract_family)
    def run_size_classification(self):
        if not all(k in self.data.columns for k in ['length', 'width', 'height']): return
        for key in ['length', 'width', 'height']: self.data[key] = pd.to_numeric(self.data[key], errors='coerce')
        self.data['volume_m3'] = (self.data['length']/1000 * self.data['width']/1000 * self.data['height']/1000)
        def classify_size(row):
            if pd.isna(row['volume_m3']): return 'Manual'
            dims = [d for d in [row['length'], row['width'], row['height']] if pd.notna(d)]
            if not dims: return 'Manual'
            max_dim = max(dims)
            if row['volume_m3'] > 1.5 or max_dim > 1200: return 'XL'
            if (0.5 < row['volume_m3'] <= 1.5) or (750 < max_dim <= 1200): return 'L'
            if (0.05 < row['volume_m3'] <= 0.5) or (150 < max_dim <= 750): return 'M'
            return 'S'
        self.data['size_classification'] = self.data.apply(classify_size, axis=1)
    def run_part_classification(self):
        if 'unit_price' not in self.data.columns or 'part_id' not in self.data.columns:
            self.data['part_classification'] = 'Manual'
            return
        self.classifier.load_data_from_dataframe(self.data)
        self.data['part_classification'] = self.classifier.classify_all_parts()
    def calculate_distances_for_location(self, current_pincode):
        current_coords = get_lat_lon(self.geolocator, current_pincode, country="India")
        if current_coords == (None, None):
            st.error(f"CRITICAL: Could not find coordinates for your pincode {current_pincode}. Distances cannot be calculated.")
            return [None] * len(self.data)
        st.success(f"Current location found at: {current_coords}")

        progress_bar = st.progress(0)
        status_text = st.empty()

        distances, distance_codes = [], []
        for col in ['pincode', 'city', 'state']:
            if col not in self.data.columns: self.data[col] = ''

        total_vendors = len(self.data)
        for i, (idx, row) in enumerate(self.data.iterrows()):
            pincode_str = str(row.get('pincode', '')).strip().split('.')[0]
            if not pincode_str or not pincode_str.isdigit() or int(pincode_str) == 0:
                distances.append(None); distance_codes.append(None)
                continue
            vendor_coords = get_lat_lon(self.geolocator, pincode_str, country="India", city=str(row.get('city', '')).strip(), state=str(row.get('state', '')).strip())
            if vendor_coords == (None, None):
                distances.append(None); distance_codes.append(None)
            else:
                try:
                    distance_km = geodesic(current_coords, vendor_coords).km
                    distances.append(distance_km)
                    distance_codes.append(get_distance_code(distance_km))
                except Exception:
                    distances.append(None); distance_codes.append(None)

            progress = (i + 1) / total_vendors
            progress_bar.progress(progress)
            status_text.text(f"Processing vendor locations: {i + 1}/{total_vendors}")

        status_text.text("Distance calculation complete.")
        return distance_codes
    def run_location_based_norms(self, location_name, pincode):
        self.data['DISTANCE CODE'] = self.calculate_distances_for_location(pincode)
        if 'part_classification' not in self.data.columns: return
        def get_inv_class(p, d):
            if pd.isna(p) or pd.isna(d): return None
            d = int(d)
            if p in ['AA', 'A']: return f"A{d}"
            if p == 'B': return f"B{d}"
            if p == 'C': return 'C1' if d in [1, 2] else 'C2'
            return None
        self.data['inventory_classification'] = self.data.apply(lambda r: get_inv_class(r.get('part_classification'), r.get('DISTANCE CODE')), axis=1)
        self.data['RM IN DAYS'] = self.data['inventory_classification'].map(self.rm_days_mapping)
        self.data['RM IN QTY'] = self.data['RM IN DAYS'] * pd.to_numeric(self.data.get('net_daily_consumption'), errors='coerce')
        self.data['RM IN INR'] = self.data['RM IN QTY'] * pd.to_numeric(self.data.get('unit_price'), errors='coerce')
        self.data['PACKING FACTOR (PF)'] = pd.to_numeric(self.data.get('packing_factor', 1), errors='coerce').fillna(1)
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        self.data['NO OF SEC. PACK REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(self.data['NO OF SEC. PACK REQD.'] * self.data['PACKING FACTOR (PF)'])
    def run_warehouse_location_assignment(self):
        if 'family' not in self.data.columns: return
        def get_wh_loc(row):
            fam, desc, vol_m3 = row.get('family', 'Others'), row.get('description', ''), row.get('volume_m3', None)
            word_match = lambda text, word: re.search(r'\b' + re.escape(word) + r'\b', str(text).upper())
            if fam == "AC": return "OUTSIDE" if word_match(desc, "BCS") else "MEZ C-02(B)"
            if fam in ["ASSY", "Bracket"] and word_match(desc, "STEERING"): return "DIRECT FROM INSTOR"
            if fam == "Bracket": return "HRR"
            if fam == "Electronics": return "CRL" if any(word_match(desc, k) for k in ["CAMERA", "APC", "MNVR", "WOODWARD"]) else "HRR"
            if fam == "Electrical": return "CRL" if vol_m3 is not None and (vol_m3 * 1_000_000) > 200 else "HRR"
            if fam == "Mechanical": return "DIRECT FROM INSTOR" if word_match(desc, "STEERING") else "HRR"
            if fam == "Plywood": return "HRR" if word_match(desc, "EDGE") else "MRR(C-01)"
            if fam == "Rubber": return "MEZ B-01" if word_match(desc, "GROMMET") else "HRR"
            if fam == "Tape": return "HRR" if word_match(desc, "BUTYL") else "MEZ B-01"
            if fam == "Wheels":
                if word_match(desc, "TYRE") and word_match(desc, "JK"): return "OUTSIDE"
                if word_match(desc, "RIM"): return "MRR(C-01)"
            return BASE_WAREHOUSE_MAPPING.get(fam, "HRR")
        self.data['wh_loc'] = self.data.apply(get_wh_loc, axis=1)


# --- 5. FINAL EXCEL REPORT GENERATION ---
@st.cache_data
def create_formatted_excel_output(_df):
    output = io.BytesIO()
    final_df = _df.copy().loc[:, ~_df.columns.duplicated()]
    enhanced_map = {'qty_veh_1': 'Qty/Veh 1', 'qty_veh_2': 'Qty/Veh 2', 'total_qty': 'TOTAL', 'qty_veh_1_daily': 'Qty/Veh 1_Daily', 'qty_veh_2_daily': 'Qty/Veh 2_Daily', **PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS, 'inventory_classification': 'INVENTORY CLASSIFICATION'}
    final_df.rename(columns={k: v for k, v in enhanced_map.items() if k in final_df.columns}, inplace=True)
    for col in [c for c in ALL_TEMPLATE_COLUMNS if c not in final_df.columns]: final_df[col] = ''
    final_df = final_df[ALL_TEMPLATE_COLUMNS]
    final_df['SR.NO'] = range(1, len(final_df) + 1)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        h_gray = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1})
        s_orange = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1})
        s_blue = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1})
        
        final_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
        worksheet = writer.sheets['Master Data Sheet']
        
        worksheet.merge_range('A1:H1', 'PART DETAILS', h_gray)
        worksheet.merge_range('I1:L1', 'Daily consumption', s_orange)
        worksheet.merge_range('M1:N1', 'PRICE & CLASSIFICATION', s_orange)
        worksheet.merge_range('O1:S1', 'Size & Classification', s_orange)
        worksheet.merge_range('T1:Z1', 'VENDOR DETAILS', s_blue)
        worksheet.merge_range('AA1:AO1', 'PACKAGING DETAILS', s_orange)
        worksheet.merge_range('AP1:AW1', 'INVENTORY NORM', s_blue)
        worksheet.merge_range('AX1:BD1', 'WH STORAGE', s_orange)
        worksheet.merge_range('BE1:BH1', 'SUPPLY SYSTEM', s_blue)
        worksheet.merge_range('BI1:BW1', 'LINE SIDE STORAGE', h_gray)

        for col_num, value in enumerate(final_df.columns): worksheet.write(1, col_num, value, h_gray)
        worksheet.set_column('A:A', 6); worksheet.set_column('B:C', 22); worksheet.set_column('D:BW', 18)
    
    return output.getvalue()

# --- 6. STREAMLIT UI ---
def manual_review_step(df, internal_key, step_name):
    st.markdown(f"""
    <div class="process-step">
        <h4>🔍 Manual Review: {step_name}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    pfep_name = INTERNAL_TO_PFEP_NEW_COLS.get(internal_key, PFEP_COLUMN_MAP.get(internal_key, internal_key))
    review_df = df[['part_id', 'description', internal_key]].copy()
    review_df.rename(columns={internal_key: pfep_name, 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION'}, inplace=True)
    
    with st.container():
        st.markdown("**📋 Preview of Classification Results:**")
        st.dataframe(review_df.head(10), use_container_width=True)
    
    csv = review_df.to_csv(index=False).encode('utf-8')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label=f"📥 Download '{step_name}' file for review",
            data=csv,
            file_name=f"manual_review_{step_name.lower().replace(' ', '_')}.csv",
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            f"📤 Upload modified '{step_name}' file (optional)", 
            type=['csv'], 
            key=f"uploader_{internal_key}"
        )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if 'PARTNO' in uploaded_df.columns and pfep_name in uploaded_df.columns:
                uploaded_df.rename(columns={pfep_name: internal_key, 'PARTNO': 'part_id'}, inplace=True)
                update_map = uploaded_df.set_index('part_id')[internal_key].to_dict()
                df[internal_key] = df['part_id'].map(update_map).fillna(df[internal_key])
                create_info_card("Success!", f"Manual changes for {step_name} applied successfully.", "success")
                return df
            else:
                create_info_card("Upload Error", "Ensure the file contains 'PARTNO' and the specific classification column.", "warning")
        except Exception as e:
            create_info_card("Processing Error", f"Error processing uploaded file: {e}", "warning")
            
    return df

def main():
    # Load custom CSS
    load_custom_css()
    
    # Page configuration
    st.set_page_config(
        layout="wide", 
        page_title="PFEP Analyzer - Modern Dashboard",
        page_icon="🏭",
        initial_sidebar_state="expanded"
    )
    
    # Custom title with gradient
    st.markdown('<h1 class="custom-title">🏭 PFEP Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Plan For Each Part - Advanced Supply Chain Analysis</p>', unsafe_allow_html=True)

    if 'processor' not in st.session_state:
        st.session_state.processor = None

    # Create process flow indicator
    create_step_indicator(1, 6)

    # --- Step 1: File Uploads and Initial Setup ---
    st.markdown('<h2 class="section-header">📂 Data Upload & Configuration</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="step-container">
            <h3>📁 Upload Your Data Files</h3>
            <p>Upload your BOM files and vendor master data to get started with the analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📊 PBOM FILES**")
            pbom_files = st.file_uploader("Upload PBOM files", accept_multiple_files=True, type=['csv', 'xlsx'], key='pbom')
            
        with col2:
            st.markdown("**📋 MBOM FILES**")
            mbom_files = st.file_uploader("Upload MBOM files", accept_multiple_files=True, type=['csv', 'xlsx'], key='mbom')
            
        with col3:
            st.markdown("**🚚 VENDOR MASTER FILE**")
            vendor_file = st.file_uploader("Upload Vendor Master file", type=['csv', 'xlsx'], key='vendor')
        
        st.markdown("""
        <div class="step-container">
            <h3>⚙️ Production Configuration</h3>
            <p>Set your daily production quantities for accurate consumption calculations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            daily_mult_1 = st.number_input("🚗 Daily production - Vehicle Type 1", min_value=0.0, value=1.0, step=0.1)
        with d_col2:
            daily_mult_2 = st.number_input("🚙 Daily production - Vehicle Type 2", min_value=0.0, value=1.0, step=0.1)

    # Prominent start button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_processing = st.button("🚀 Start Data Processing", key="start_btn", help="Begin the comprehensive PFEP analysis", use_container_width=True)
        st.markdown('<style>.stButton > button { font-size: 1.2rem !important; padding: 0.8rem 2rem !important; }</style>', unsafe_allow_html=True)

    if start_processing:
        if not pbom_files and not mbom_files:
            create_info_card("No Data Found", "Please upload at least one PBOM or MBOM file to proceed.", "warning")
            return

        with st.spinner("🔄 Consolidating and processing your data..."):
            bom_files = {"PBOM": pbom_files, "MBOM": mbom_files}
            master_df, logs = consolidate_data(bom_files, vendor_file, daily_mult_1, daily_mult_2)
            
            # Display consolidation results
            st.markdown('<h3 class="section-header">📊 Processing Results</h3>', unsafe_allow_html=True)
            
            with st.expander("📋 View Processing Log", expanded=False):
                for log in logs:
                    if "✅" in log:
                        st.success(log)
                    elif "⚠️" in log:
                        st.warning(log)
                    elif "❌" in log:
                        st.error(log)
                    else:
                        st.info(log)

            if master_df is not None:
                create_info_card("Data Processing Complete", f"Successfully processed {len(master_df)} unique parts. Ready for classification steps.", "success")
                st.session_state.processor = ComprehensiveInventoryProcessor(master_df.loc[:, ~master_df.columns.duplicated()])
                
                # Display summary metrics
                total_parts = len(master_df)
                total_consumption = master_df.get('net_daily_consumption', pd.Series([0])).sum()
                avg_price = master_df.get('unit_price', pd.Series([0])).mean()
                
                create_metric_display([
                    ("Total Parts", f"{total_parts:,}", None),
                    ("Daily Consumption", f"{total_consumption:,.1f}", None),
                    ("Avg Unit Price", f"₹{avg_price:,.2f}", None)
                ])
                
                with st.expander("👀 Preview Processed Data", expanded=False):
                    st.dataframe(master_df.head(20), use_container_width=True)
            else:
                create_info_card("Processing Failed", "Data consolidation failed. Please check the processing log above.", "warning")

    if st.session_state.processor:
        processor = st.session_state.processor

        # Update process flow
        create_step_indicator(2, 6)

        # --- Processing Steps with Modern UI ---
        st.markdown('<h2 class="section-header">⚙️ Processing Pipeline</h2>', unsafe_allow_html=True)
        
        # Step 1: Family Classification
        with st.container():
            st.markdown("""
            <div class="process-step">
                <h3>🏷️ Step 1: Family Classification</h3>
                <p>Automatically categorize parts based on keywords in descriptions</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🔄 Run Family Classification", key="family_btn", use_container_width=True):
                    with st.spinner("Analyzing part descriptions..."):
                        processor.run_family_classification()
                        create_info_card("Classification Complete", "Family classification has been completed successfully.", "success")
            
            with col2:
                if 'family' in processor.data.columns:
                    families = processor.data['family'].value_counts()
                    st.metric("Families Found", len(families))
            
            if 'family' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'family', 'Family Classification')
        
        # Step 2: Size Classification
        with st.container():
            st.markdown("""
            <div class="process-step">
                <h3>📏 Step 2: Size Classification</h3>
                <p>Classify parts by physical dimensions and volume calculations</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🔄 Run Size Classification", key="size_btn", use_container_width=True):
                    with st.spinner("Calculating volumes and dimensions..."):
                        processor.run_size_classification()
                        create_info_card("Classification Complete", "Size classification has been completed successfully.", "success")
            
            with col2:
                if 'size_classification' in processor.data.columns:
                    sizes = processor.data['size_classification'].value_counts()
                    st.metric("Size Categories", len(sizes))
            
            if 'size_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'size_classification', 'Size Classification')

        # Step 3: Part Classification
        with st.container():
            st.markdown("""
            <div class="process-step">
                <h3>💰 Step 3: Part Classification (ABC Analysis)</h3>
                <p>Percentage-based classification using unit price analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🔄 Run Part Classification", key="part_btn", use_container_width=True):
                    with st.spinner("Performing ABC analysis..."):
                        processor.run_part_classification()
                        create_info_card("Classification Complete", "Part classification has been completed successfully.", "success")
                        
                        # Display classification ranges
                        st.markdown("**📊 Classification Ranges:**")
                        for class_name, info in processor.classifier.calculated_ranges.items():
                            if info['min'] is not None:
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric(f"{class_name} Class", f"{info['count']} parts")
                                with col_b:
                                    st.metric("Min Price", f"₹{info['min']:,.2f}")
                                with col_c:
                                    st.metric("Max Price", f"₹{info['max']:,.2f}")
            
            with col2:
                if 'part_classification' in processor.data.columns:
                    classes = processor.data['part_classification'].value_counts()
                    st.metric("Classification Types", len(classes))
            
            if 'part_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'part_classification', 'Part Classification')

        # Step 4: Distance & Inventory Norms
        with st.container():
            st.markdown("""
            <div class="process-step">
                <h3>🗺️ Step 4: Distance & Inventory Norms</h3>
                <p>Calculate vendor distances and determine inventory requirements</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                current_pincode = st.text_input("📍 Your facility pincode", value="411001", help="Enter your current location pincode for distance calculations")
                
                if st.button("🔄 Calculate Inventory Norms", key="distance_btn", use_container_width=True):
                    with st.spinner("🌍 Calculating distances and inventory norms... This may take several minutes."):
                        processor.run_location_based_norms("Current Location", current_pincode)
                        create_info_card("Calculation Complete", "Distance calculations and inventory norms have been established.", "success")
            
            with col2:
                if 'DISTANCE CODE' in processor.data.columns:
                    distances = processor.data['DISTANCE CODE'].value_counts()
                    st.metric("Distance Codes", len(distances))
            
            if 'inventory_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'inventory_classification', 'Inventory Classification')

        # Step 5: Warehouse Location Assignment
        with st.container():
            st.markdown("""
            <div class="process-step">
                <h3>🏪 Step 5: Warehouse Location Assignment</h3>
                <p>Assign optimal warehouse locations based on part characteristics</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🔄 Assign Warehouse Locations", key="warehouse_btn", use_container_width=True):
                    with st.spinner("Optimizing warehouse assignments..."):
                        processor.run_warehouse_location_assignment()
                        create_info_card("Assignment Complete", "Warehouse locations have been assigned successfully.", "success")
            
            with col2:
                if 'wh_loc' in processor.data.columns:
                    locations = processor.data['wh_loc'].value_counts()
                    st.metric("Warehouse Locations", len(locations))
            
            if 'wh_loc' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'wh_loc', 'Warehouse Location')

        # Final step indicator
        create_step_indicator(6, 6)

        # Step 6: Final Report Generation
        st.markdown('<h2 class="section-header">📊 Final Report Generation</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown("""
            <div class="step-container">
                <h3>🎉 Generate Comprehensive PFEP Report</h3>
                <p>Create your final formatted Excel report with all classifications and analysis results.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Final summary metrics
            if hasattr(processor, 'data'):
                summary_data = processor.data
                
                # Calculate completion metrics
                family_complete = 'family' in summary_data.columns
                size_complete = 'size_classification' in summary_data.columns
                part_complete = 'part_classification' in summary_data.columns
                distance_complete = 'inventory_classification' in summary_data.columns
                warehouse_complete = 'wh_loc' in summary_data.columns
                
                completion_count = sum([family_complete, size_complete, part_complete, distance_complete, warehouse_complete])
                completion_percent = (completion_count / 5) * 100
                
                # Display completion status
                st.markdown("**🎯 Processing Status:**")
                progress_col1, progress_col2 = st.columns([3, 1])
                
                with progress_col1:
                    st.progress(completion_percent / 100)
                with progress_col2:
                    st.metric("Completion", f"{completion_percent:.0f}%")
                
                # Status indicators for each step
                status_cols = st.columns(5)
                statuses = [
                    ("Family", family_complete),
                    ("Size", size_complete), 
                    ("Part Class", part_complete),
                    ("Inventory", distance_complete),
                    ("Warehouse", warehouse_complete)
                ]
                
                for i, (name, complete) in enumerate(statuses):
                    with status_cols[i]:
                        status_icon = "✅" if complete else "⏳"
                        st.markdown(f"**{status_icon} {name}**")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Generate report button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📥 Generate Final Excel Report", key="final_report_btn", use_container_width=True):
                        with st.spinner("🔄 Generating your comprehensive Excel report..."):
                            excel_data = create_formatted_excel_output(processor.data)
                            
                            # Success animation and download
                            st.balloons()
                            create_info_card("Report Ready!", "Your comprehensive PFEP analysis report has been generated successfully.", "success")
                            
                            # Download button with custom styling
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_data,
                                file_name="PFEP_Analysis_Report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                
                # Final analytics summary
                if completion_percent > 50:
                    st.markdown('<h3 class="section-header">📈 Analysis Summary</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_parts = len(summary_data)
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Total Parts Analyzed</h4>
                            <h2>{total_parts:,}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if 'family' in summary_data.columns:
                            unique_families = summary_data['family'].nunique()
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Part Families</h4>
                                <h2>{unique_families}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        if 'wh_loc' in summary_data.columns:
                            warehouse_locations = summary_data['wh_loc'].nunique()
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Warehouse Locations</h4>
                                <h2>{warehouse_locations}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col4:
                        if 'net_daily_consumption' in summary_data.columns:
                            total_consumption = summary_data['net_daily_consumption'].sum()
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Daily Consumption</h4>
                                <h2>{total_consumption:,.0f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Distribution charts if data is available
                    if family_complete:
                        with st.expander("📊 Family Distribution Analysis", expanded=False):
                            family_counts = summary_data['family'].value_counts().head(10)
                            st.bar_chart(family_counts)
                    
                    if part_complete:
                        with st.expander("💰 Part Classification Distribution", expanded=False):
                            class_counts = summary_data['part_classification'].value_counts()
                            st.bar_chart(class_counts)

        # Footer
        st.markdown("""
        <div class="footer">
            <h3>🎉 PFEP Analysis Complete!</h3>
            <p>Your comprehensive Plan For Each Part analysis has been completed successfully.</p>
            <p><strong>Next Steps:</strong> Review your Excel report and implement the warehouse location and inventory recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
