import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Inventory & Supply Chain Analysis Tool",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS AND MAPPINGS ---

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

PFEP_COLUMN_MAP = {
    'part_id': 'PARTNO',
    'description': 'PART DESCRIPTION',
    'qty_veh': 'Qty/Veh',
    'qty/veh': 'Qty/Veh',
    'quantity_per_vehicle': 'Qty/Veh',
    'net_daily_consumption': 'NET',
    'unit_price': 'UNIT PRICE',
    'vendor_code': 'VENDOR CODE',
    'vendor_name': 'VENDOR NAME',
    'city': 'CITY',
    'state': 'STATE',
    'country': 'COUNTRY',
    'pincode': 'PINCODE',
    'length': 'L-MM_Size',
    'width': 'W-MM_Size',
    'height': 'H-MM_Size',
    'qty_per_pack': 'QTY/PACK_Sec',
    'packing_factor': 'PACKING FACTOR (PF)',
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

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'geocoding_cache' not in st.session_state:
    st.session_state.geocoding_cache = {}
if 'processing_step' not in st.session_state:
    st.session_state.processing_step = 0

# --- UTILITY FUNCTIONS ---

@st.cache_data
def get_lat_lon(pincode, country="India", city="", state="", retries=3):
    """Geocodes a location using its pincode."""
    pincode_str = str(pincode).strip().split('.')[0]
    if not pincode_str.isdigit() or int(pincode_str) == 0:
        return (None, None)
    
    query_key = f"{pincode_str}|{country}"
    
    if query_key in st.session_state.geocoding_cache:
        return st.session_state.geocoding_cache[query_key]
    
    geolocator = Nominatim(user_agent="inventory_distance_calculator_streamlit", timeout=10)
    
    if city and state:
        query = f"{pincode_str}, {city}, {state}, {country}"
    else:
        query = f"{pincode_str}, {country}"
    
    for attempt in range(retries):
        try:
            time.sleep(1)
            location = geolocator.geocode(query)
            if location:
                coords = (location.latitude, location.longitude)
                st.session_state.geocoding_cache[query_key] = coords
                return coords
        except Exception as e:
            if attempt < retries - 1:
                continue
    
    st.session_state.geocoding_cache[query_key] = (None, None)
    return (None, None)

def get_distance_code(distance):
    """Classifies distance into codes."""
    if pd.isna(distance):
        return None
    elif distance < 50:
        return 1
    elif distance <= 250:
        return 2
    elif distance <= 750:
        return 3
    else:
        return 4

def find_qty_veh_column(df):
    """Find the qty/veh column in the DataFrame."""
    possible_names = [
        'qty/veh', 'Qty/Veh', 'QTY/VEH', 'qty_veh', 'Qty_Veh',
        'quantity/vehicle', 'Quantity/Vehicle', 'QUANTITY/VEHICLE',
        'qty per veh', 'Qty Per Veh', 'QTY PER VEH',
        'vehicle qty', 'Vehicle Qty', 'VEHICLE QTY'
    ]
    
    for col in df.columns:
        if col in possible_names:
            return col
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for possible in possible_names:
            if col_lower == possible.lower():
                return col
    return None

def find_and_rename_columns(df, file_number=None):
    """Automatically finds and renames columns."""
    rename_dict = {}
    found_keys = []
    
    qty_veh_col = find_qty_veh_column(df)
    if qty_veh_col:
        if file_number == 1:
            rename_dict[qty_veh_col] = 'qty_veh_1'
            found_keys.append('qty_veh_1')
        elif file_number == 2:
            rename_dict[qty_veh_col] = 'qty_veh_2'
            found_keys.append('qty_veh_2')
        else:
            rename_dict[qty_veh_col] = 'qty_veh'
            found_keys.append('qty_veh')
    
    for internal_key, pfep_name in PFEP_COLUMN_MAP.items():
        if internal_key in ['qty_veh', 'qty/veh', 'quantity_per_vehicle']:
            continue
        if pfep_name in df.columns:
            rename_dict[pfep_name] = internal_key
            found_keys.append(internal_key)
    
    df.rename(columns=rename_dict, inplace=True)
    return df, 'part_id' in found_keys

# --- CLASSIFICATION CLASSES ---

class PartClassificationSystem:
    def __init__(self):
        self.percentages = {'C': {'target': 60, 'tolerance': 5}, 'B': {'target': 25, 'tolerance': 2}, 'A': {'target': 12, 'tolerance': 2}, 'AA': {'target': 3, 'tolerance': 1}}
        self.calculated_ranges = {}
        self.parts_data = None
    
    def load_data_from_dataframe(self, df, price_column='unit_price', part_id_column='part_id'):
        self.parts_data = df.copy()
        if price_column not in self.parts_data.columns:
            raise ValueError(f"Column '{price_column}' not found")
        if part_id_column not in self.parts_data.columns:
            raise ValueError(f"Column '{part_id_column}' not found")
        self.price_column = price_column
        self.part_id_column = part_id_column
        self.calculate_percentage_ranges()
    
    def calculate_percentage_ranges(self):
        if self.parts_data is None:
            return
        
        valid_prices = [float(p) for p in self.parts_data[self.price_column] if not self.is_blank_price(p)]
        if not valid_prices:
            return
        
        valid_prices = sorted(valid_prices)
        total_valid_parts = len(valid_prices)
        
        ranges = {}
        cumulative_percent = 0
        
        for class_name in ['C', 'B', 'A', 'AA']:
            target_percent = self.percentages[class_name]['target']
            start_idx = int((cumulative_percent / 100) * total_valid_parts)
            end_idx = int(((cumulative_percent + target_percent) / 100) * total_valid_parts) - 1
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, total_valid_parts - 1)
            
            if start_idx <= end_idx:
                ranges[class_name] = {
                    'min': valid_prices[start_idx],
                    'max': valid_prices[end_idx],
                    'actual_count': end_idx - start_idx + 1,
                    'actual_percent': round((end_idx - start_idx + 1) / total_valid_parts * 100, 2),
                    'target_percent': target_percent,
                    'tolerance': f"±{self.percentages[class_name]['tolerance']}%"
                }
            else:
                ranges[class_name] = {
                    'min': None, 'max': None, 'actual_count': 0,
                    'actual_percent': 0, 'target_percent': target_percent,
                    'tolerance': f"±{self.percentages[class_name]['tolerance']}%"
                }
            cumulative_percent += target_percent
        
        self.calculated_ranges = ranges
    
    def is_blank_price(self, unit_price):
        return pd.isna(unit_price) or str(unit_price).strip().lower() in ['', 'nan', 'null', 'none', 'n/a']
    
    def classify_part(self, unit_price, part_id=None):
        if self.is_blank_price(unit_price):
            return 'Manual'
        
        try:
            unit_price = float(unit_price)
        except (ValueError, TypeError):
            return 'Manual'
        
        for class_name in ['AA', 'A', 'B', 'C']:
            range_info = self.calculated_ranges.get(class_name)
            if range_info and range_info['min'] is not None:
                if range_info['min'] <= unit_price <= range_info['max']:
                    return class_name
        
        return 'Unclassified'
    
    def classify_all_parts(self):
        if self.parts_data is None:
            return None
        
        if not self.calculated_ranges:
            self.calculate_percentage_ranges()
        
        if not self.calculated_ranges:
            return None
        
        classified_data = self.parts_data.copy()
        classified_data['classification'] = classified_data.apply(
            lambda r: self.classify_part(r[self.price_column], r[self.part_id_column]), axis=1
        )
        return classified_data

class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data
        self.rm_days_mapping = {
            'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11,
            'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16,
            'C1': 16, 'C2': 31
        }
        self.classifier = PartClassificationSystem()
    
    def run_family_classification(self):
        if 'description' not in self.data.columns:
            return
        
        def find_kw_pos(desc, kw):
            match = re.search(r'\b' + re.escape(str(kw).upper()) + r'\b', str(desc).upper())
            return match.start() if match else -1
        
        def extract_family(desc):
            if pd.isna(desc):
                return 'Others'
            
            # Priority families first
            for fam in CATEGORY_PRIORITY_FAMILIES:
                if fam in FAMILY_KEYWORD_MAPPING:
                    if any(find_kw_pos(desc, kw) != -1 for kw in FAMILY_KEYWORD_MAPPING[fam]):
                        return fam
            
            # Other families
            matches = []
            for fam, kws in FAMILY_KEYWORD_MAPPING.items():
                if fam not in CATEGORY_PRIORITY_FAMILIES:
                    for kw in kws:
                        pos = find_kw_pos(desc, kw)
                        if pos != -1:
                            matches.append((pos, fam))
            
            return min(matches, key=lambda x: x[0])[1] if matches else 'Others'
        
        self.data['family'] = self.data['description'].apply(extract_family)
    
    def run_size_classification(self):
        if not all(k in self.data.columns for k in ['length', 'width', 'height']):
            return
        
        for key in ['length', 'width', 'height']:
            self.data[key] = pd.to_numeric(self.data[key], errors='coerce')
        
        self.data['volume_m3'] = self.data.apply(
            lambda r: (r['length']/1000 * r['width']/1000 * r['height']/1000)
            if pd.notna(r['length']) and pd.notna(r['width']) and pd.notna(r['height'])
            else None, axis=1
        )
        
        def classify_size(row):
            if pd.isna(row['volume_m3']):
                return 'Manual'
            
            dims = [d for d in [row['length'], row['width'], row['height']] if pd.notna(d)]
            if not dims:
                return 'Manual'
            
            max_dim = max(dims)
            
            if row['volume_m3'] > 1.5 or max_dim > 1200:
                return 'XL'
            elif (0.5 < row['volume_m3'] <= 1.5) or (750 < max_dim <= 1200):
                return 'L'
            elif (0.05 < row['volume_m3'] <= 0.5) or (150 < max_dim <= 750):
                return 'M'
            else:
                return 'S'
        
        self.data['size_classification'] = self.data.apply(classify_size, axis=1)
    
    def run_part_classification(self):
        if 'unit_price' not in self.data.columns or 'part_id' not in self.data.columns:
            self.data['part_classification'] = 'Manual'
            return
        
        self.classifier.load_data_from_dataframe(self.data)
        classified_df = self.classifier.classify_all_parts()
        
        if classified_df is not None:
            self.data['part_classification'] = self.data['part_id'].map(
                classified_df.set_index('part_id')['classification'].to_dict()
            )
        else:
            self.data['part_classification'] = 'Manual'
    
    def calculate_distances_for_location(self, current_pincode):
        current_coords = get_lat_lon(current_pincode, country="India")
        if current_coords == (None, None):
            return [None] * len(self.data)
        
        for col in ['pincode', 'city', 'state']:
            if col not in self.data.columns:
                self.data[col] = ''
        
        distances = []
        distance_codes = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_vendors = len(self.data)
        
        for idx, row in self.data.iterrows():
            progress = (idx + 1) / total_vendors
            progress_bar.progress(progress)
            status_text.text(f"Processing vendor locations... {idx + 1}/{total_vendors}")
            
            vendor_pincode = row.get('pincode', '')
            pincode_str = str(vendor_pincode).strip().split('.')[0]
            
            if not pincode_str or pincode_str.lower() in ['nan', 'none', 'na', 'n/a', '0']:
                distances.append(None)
                distance_codes.append(None)
                continue
            
            vendor_coords = get_lat_lon(
                pincode_str, 
                country="India", 
                city=str(row.get('city', '')).strip(), 
                state=str(row.get('state', '')).strip()
            )
            
            if vendor_coords == (None, None):
                distances.append(None)
                distance_codes.append(None)
            else:
                try:
                    distance_km = geodesic(current_coords, vendor_coords).km
                    distances.append(distance_km)
                    distance_codes.append(get_distance_code(distance_km))
                except Exception:
                    distances.append(None)
                    distance_codes.append(None)
        
        progress_bar.empty()
        status_text.empty()
        
        return distance_codes
    
    def run_location_based_norms(self, location_name, pincode):
        distance_codes = self.calculate_distances_for_location(pincode)
        distance_code_col = 'DISTANCE CODE'
        self.data[distance_code_col] = distance_codes
        
        if 'part_classification' not in self.data.columns:
            return
        
        def get_inv_class(p, d):
            if pd.isna(p) or pd.isna(d):
                return None
            d = int(d)
            if p in ['AA', 'A']:
                return f"A{d}"
            elif p == 'B':
                return f"B{d}"
            elif p == 'C':
                return 'C1' if d in [1, 2] else 'C2'
            return None
        
        inv_class_col_internal = 'inventory_classification'
        inv_class_col_pfep = 'INVENTORY CLASSIFICATION'
        
        self.data[inv_class_col_internal] = self.data.apply(
            lambda r: get_inv_class(r.get('part_classification'), r.get(distance_code_col)), axis=1
        )
        
        INTERNAL_TO_PFEP_NEW_COLS[inv_class_col_internal] = inv_class_col_pfep
        
        self.data['RM IN DAYS'] = self.data[inv_class_col_internal].map(self.rm_days_mapping)
        self.data['RM IN QTY'] = self.data['RM IN DAYS'] * pd.to_numeric(
            self.data.get('net_daily_consumption'), errors='coerce'
        )
        self.data['RM IN INR'] = self.data['RM IN QTY'] * pd.to_numeric(
            self.data.get('unit_price'), errors='coerce'
        )
        
        self.data['PACKING FACTOR (PF)'] = pd.to_numeric(
            self.data.get('packing_factor', 1), errors='coerce'
        ).fillna(1)
        
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        
        self.data['NO OF SEC. PACK REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(
            self.data['NO OF SEC. PACK REQD.'] * self.data['PACKING FACTOR (PF)']
        )
    
    def run_warehouse_location_assignment(self):
        if 'family' not in self.data.columns:
            return
        
        def get_wh_loc(row):
            fam = row.get('family', 'Others')
            desc = row.get('description', '')
            vol_m3 = row.get('volume_m3', None)
            
            def word_match(text, word):
                return re.search(r'\b' + re.escape(word) + r'\b', str(text).upper())
            
            # Special rules
            if fam == "AC":
                return "OUTSIDE" if word_match(desc, "BCS") else "MEZ C-02(B)"
            
            if fam in ["ASSY", "Bracket"] and word_match(desc, "STEERING"):
                return "DIRECT FROM INSTOR"
            
            if fam == "Bracket":
                return "HRR"
            
            if fam == "Electronics":
                return "CRL" if any(word_match(desc, k) for k in ["CAMERA", "APC", "MNVR", "WOODWARD"]) else "HRR"
            
            if fam == "Electrical":
                return "CRL" if vol_m3 is not None and (vol_m3 * 1_000_000) > 200 else "HRR"
            
            if fam == "Mechanical":
                return "DIRECT FROM INSTOR" if word_match(desc, "STEERING") else "HRR"
            
            if fam == "Plywood":
                return "HRR" if word_match(desc, "EDGE") else "MRR(C-01)"
            
            if fam == "Rubber":
                return "MEZ B-01" if word_match(desc, "GROMMET") else "HRR"
            
            if fam == "Tape":
                return "HRR" if word_match(desc, "BUTYL") else "MEZ B-01"
            
            if fam == "Wheels":
                if word_match(desc, "TYRE") and word_match(desc, "JK"):
                    return "OUTSIDE"
                if word_match(desc, "RIM"):
                    return "MRR(C-01)"
            
            return BASE_WAREHOUSE_MAPPING.get(fam, "HRR")
        
        self.data['wh_loc'] = self.data.apply(get_wh_loc, axis=1)

# --- STREAMLIT APP ---

def main():
    st.title("🏭 Inventory & Supply Chain Analysis Tool")
    st.markdown("**Comprehensive PFEP Analysis System**")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Data Upload & Processing",
        "📈 Analysis Dashboard",
        "📋 Results & Export"
    ])
    
    if page == "📊 Data Upload & Processing":
        show_upload_page()
    elif page == "📈 Analysis Dashboard":
        show_dashboard()
    elif page == "📋 Results & Export":
        show_results_page()

def show_upload_page():
    st.header("📊 Data Upload & Processing")
    
    # Step 1: File Upload
    st.subheader("Step 1: Upload BOM Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PBOM Files**")
        pbom_files = st.file_uploader(
            "Upload PBOM files", 
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            key="pbom_files"
        )
    
    with col2:
        st.markdown("**MBOM Files**")
        mbom_files = st.file_uploader(
            "Upload MBOM files", 
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            key="mbom_files"
        )
    
    # Vendor Master File
    st.subheader("Step 2: Upload Vendor Master")
    st.markdown("Upload your vendor master file containing vendor details, locations, and pricing information")
    vendor_file = st.file_uploader(
        "Choose Vendor Master file", 
        type=['csv', 'xlsx', 'xls'],
        key="vendor_file",
        help="This file should contain vendor information like codes, names, locations, pincodes, etc."
    )
    
    if vendor_file:
        st.success(f"✅ Vendor file selected: {vendor_file.name}")
    
    # Daily Consumption Parameters
    st.subheader("Step 3: Daily Consumption Parameters")
    st.markdown("Enter the daily production quantities for different vehicle types to calculate daily consumption")
    
    col1, col2 = st.columns(2)
    with col1:
        daily_qty_1 = st.number_input(
            "Daily production quantity for Vehicle Type 1:", 
            value=1.0, 
            min_value=0.0, 
            step=0.1,
            help="Enter the number of Vehicle Type 1 produced per day"
        )
    with col2:
        daily_qty_2 = st.number_input(
            "Daily production quantity for Vehicle Type 2:", 
            value=1.0, 
            min_value=0.0, 
            step=0.1,
            help="Enter the number of Vehicle Type 2 produced per day"
        )
    
    if daily_qty_1 != 1.0 or daily_qty_2 != 1.0:
        st.info(f"📈 Daily consumption will be calculated as: Qty/Veh × Daily Production Quantity")
        st.write(f"   • Vehicle Type 1: Qty/Veh × {daily_qty_1}")
        st.write(f"   • Vehicle Type 2: Qty/Veh × {daily_qty_2}")
    
    # Location Settings
    st.subheader("Step 4: Location Settings")
    st.markdown("Enter your facility's pincode to calculate distances to vendor locations")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        current_pincode = st.text_input(
            "Enter your current pincode for distance calculation:", 
            value="411001",
            help="This will be used to calculate distances to all vendor locations"
        )
    with col2:
        if current_pincode:
            st.info(f"📍 Using pincode: **{current_pincode}**")
    
    # Validation before processing
    st.subheader("Step 5: Process Data")
    
    # Check if minimum requirements are met
    can_process = (pbom_files or mbom_files) and current_pincode
    
    if not (pbom_files or mbom_files):
        st.error("❌ Please upload at least one BOM file (PBOM or MBOM) to proceed.")
    elif not current_pincode:
        st.error("❌ Please enter a pincode for distance calculations.")
    else:
        st.success("✅ All required data provided. Ready to process!")
        
        # Show processing summary
        with st.expander("📋 Processing Summary", expanded=True):
            st.markdown("**Files to be processed:**")
            if pbom_files:
                st.write(f"• **{len(pbom_files)} PBOM file(s)**")
            if mbom_files:
                st.write(f"• **{len(mbom_files)} MBOM file(s)**")
            if vendor_file:
                st.write(f"• **1 Vendor Master file**")
            
            st.markdown("**Processing steps:**")
            st.write("1. 🏷️ Family Classification")
            st.write("2. 📏 Size Classification") 
            st.write("3. 💰 Part Classification (ABC Analysis)")
            st.write("4. 📍 Distance Calculation & Inventory Norms")
            st.write("5. 🏪 Warehouse Location Assignment")
            
            st.markdown("**Configuration:**")
            st.write(f"• Daily Qty Vehicle 1: {daily_qty_1}")
            st.write(f"• Daily Qty Vehicle 2: {daily_qty_2}")
            st.write(f"• Current Location Pincode: {current_pincode}")
    
    # Process Button
    if st.button("🚀 Start Processing", type="primary", disabled=not can_process):
        try:
            # Initialize progress tracking
            st.markdown("---")
            st.subheader("🔄 Processing Status")
            
            # Process uploaded files
            all_boms = []
            file_counter = 0
            
            with st.spinner("Reading and validating uploaded files..."):
                # Process PBOM files
                if pbom_files:
                    st.write("**Processing PBOM files:**")
                    for i, file in enumerate(pbom_files, 1):
                        with st.spinner(f"Processing PBOM file {i}/{len(pbom_files)}: {file.name}"):
                            df = read_uploaded_file(file)
                            if df is not None:
                                file_counter += 1
                                df, has_part_id = find_and_rename_columns(df)
                                if has_part_id:
                                    all_boms.append(df)
                                    st.success(f"✅ PBOM {i}: Added {len(df)} records from '{file.name}'")
                                else:
                                    st.warning(f"⚠️ PBOM {i}: Part ID column not found in '{file.name}'. Skipping file.")
                            else:
                                st.error(f"❌ PBOM {i}: Failed to read '{file.name}'")
                
                # Process MBOM files
                if mbom_files:
                    st.write("**Processing MBOM files:**")
                    for i, file in enumerate(mbom_files, 1):
                        with st.spinner(f"Processing MBOM file {i}/{len(mbom_files)}: {file.name}"):
                            df = read_uploaded_file(file)
                            if df is not None:
                                file_counter += 1
                                # Pass the file number for MBOM files to handle different vehicle types
                                df, has_part_id = find_and_rename_columns(df, i)
                                if has_part_id:
                                    all_boms.append(df)
                                    st.success(f"✅ MBOM {i}: Added {len(df)} records from '{file.name}'")
                                else:
                                    st.warning(f"⚠️ MBOM {i}: Part ID column not found in '{file.name}'. Skipping file.")
                            else:
                                st.error(f"❌ MBOM {i}: Failed to read '{file.name}'")
            
            if not all_boms:
                st.error("❌ No valid BOM data loaded from any file. Cannot proceed.")
                return
            
            st.success(f"✅ Successfully loaded {len(all_boms)} BOM files with data")
            
            # Consolidate data
            with st.spinner("Consolidating BOM data from multiple files..."):
                master_bom = consolidate_boms(all_boms)
                master_bom = process_qty_columns(master_bom, daily_qty_1, daily_qty_2)
                st.success(f"✅ Consolidated data: {len(master_bom)} unique parts")
            
            # Process vendor data
            if vendor_file:
                with st.spinner("Processing and merging vendor data..."):
                    vendor_df = read_uploaded_file(vendor_file)
                    if vendor_df is not None:
                        vendor_df, has_part_id = find_and_rename_columns(vendor_df)
                        if has_part_id:
                            before_merge = len(master_bom)
                            master_bom = merge_vendor_data(master_bom, vendor_df)
                            st.success(f"✅ Vendor data merged successfully. Records: {before_merge} → {len(master_bom)}")
                        else:
                            st.warning("⚠️ Part ID not found in vendor file. Cannot merge vendor data.")
                    else:
                        st.error("❌ Failed to read vendor file.")
            
            # Initialize processor
            processor = ComprehensiveInventoryProcessor(master_bom)
            
            # Show processing progress
            st.markdown("---")
            st.subheader("📊 Data Processing Steps")
            
            # Create a progress tracking container
            progress_container = st.container()
            
            with progress_container:
                st.subheader("Processing Steps")
                
                # Step 1: Family Classification
                with st.expander("Step 1: Family Classification", expanded=True):
                    with st.spinner("Running family classification..."):
                        processor.run_family_classification()
                    st.success("✅ Family classification complete")
                    
                    if 'family' in processor.data.columns:
                        family_counts = processor.data['family'].value_counts()
                        fig = px.bar(x=family_counts.index, y=family_counts.values,
                                   title="Family Distribution")
                        fig.update_layout(xaxis_title="Family", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Step 2: Size Classification
                with st.expander("Step 2: Size Classification", expanded=True):
                    with st.spinner("Running size classification..."):
                        processor.run_size_classification()
                    st.success("✅ Size classification complete")
                    
                    if 'size_classification' in processor.data.columns:
                        size_counts = processor.data['size_classification'].value_counts()
                        fig = px.pie(values=size_counts.values, names=size_counts.index,
                                   title="Size Classification Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Step 3: Part Classification
                with st.expander("Step 3: Part Classification", expanded=True):
                    with st.spinner("Running part classification..."):
                        processor.run_part_classification()
                    st.success("✅ Part classification complete")
                    
                    if hasattr(processor.classifier, 'calculated_ranges') and processor.classifier.calculated_ranges:
                        st.markdown("**Classification Ranges:**")
                        for class_name in ['AA', 'A', 'B', 'C']:
                            range_info = processor.classifier.calculated_ranges.get(class_name)
                            if range_info and range_info['min'] is not None:
                                st.write(f"**{class_name} Class:** {range_info['actual_percent']}% "
                                        f"({range_info['actual_count']} parts) - "
                                        f"${range_info['min']:,.2f} to ${range_info['max']:,.2f}")
                        
                        if 'part_classification' in processor.data.columns:
                            part_counts = processor.data['part_classification'].value_counts()
                            fig = px.bar(x=part_counts.index, y=part_counts.values,
                                       title="Part Classification Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Step 4: Location-based Processing
                with st.expander("Step 4: Distance & Inventory Norms", expanded=True):
                    with st.spinner("Calculating distances and inventory norms..."):
                        processor.run_location_based_norms("Current Location", current_pincode)
                    st.success("✅ Distance and inventory norms calculated")
                    
                    # Show distance distribution
                    if 'DISTANCE CODE' in processor.data.columns:
                        distance_counts = processor.data['DISTANCE CODE'].value_counts().sort_index()
                        distance_labels = {
                            1: "< 50 Km", 2: "50-250 Km", 
                            3: "250-750 Km", 4: "> 750 Km"
                        }
                        
                        fig = px.bar(
                            x=[distance_labels.get(i, f"Code {i}") for i in distance_counts.index],
                            y=distance_counts.values,
                            title="Distance Code Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Step 5: Warehouse Location Assignment
                with st.expander("Step 5: Warehouse Location Assignment", expanded=True):
                    with st.spinner("Assigning warehouse locations..."):
                        processor.run_warehouse_location_assignment()
                    st.success("✅ Warehouse location assignment complete")
                    
                    if 'wh_loc' in processor.data.columns:
                        wh_counts = processor.data['wh_loc'].value_counts()
                        fig = px.pie(values=wh_counts.values, names=wh_counts.index,
                                   title="Warehouse Location Distribution")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Store processed data in session state
            st.session_state.processed_data = processor.data
            st.session_state.processor = processor
            
            st.success("🎉 Processing complete! Navigate to the Analysis Dashboard or Results page.")
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)

def read_uploaded_file(file):
    """Read uploaded file into DataFrame"""
    try:
        if file.name.lower().endswith('.csv'):
            return pd.read_csv(file, low_memory=False)
        elif file.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            st.error(f"Unsupported file type: {file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading file {file.name}: {str(e)}")
        return None

def consolidate_boms(all_boms):
    """Consolidate multiple BOM DataFrames"""
    if not all_boms:
        return None
    
    master_bom = all_boms[0].copy()
    
    for i, df in enumerate(all_boms[1:], 2):
        overlap_cols = set(master_bom.columns) & set(df.columns) - {'part_id'}
        
        if overlap_cols:
            master_bom = pd.merge(master_bom, df, on='part_id', how='outer', suffixes=('', f'_temp{i}'))
            for col in overlap_cols:
                temp_col = f"{col}_temp{i}"
                if temp_col in master_bom.columns:
                    master_bom[col] = master_bom[col].fillna(master_bom[temp_col])
                    master_bom.drop(columns=[temp_col], inplace=True)
        else:
            master_bom = pd.merge(master_bom, df, on='part_id', how='outer')
    
    return master_bom

def process_qty_columns(df, daily_mult_1, daily_mult_2):
    """Process and clean quantity columns"""
    # Create qty_veh columns if they don't exist
    if 'qty_veh_1' not in df.columns:
        df['qty_veh_1'] = 0
    if 'qty_veh_2' not in df.columns:
        df['qty_veh_2'] = 0
    
    # Clean numeric columns
    df['qty_veh_1'] = pd.to_numeric(df['qty_veh_1'], errors='coerce').fillna(0)
    df['qty_veh_2'] = pd.to_numeric(df['qty_veh_2'], errors='coerce').fillna(0)
    
    # Calculate totals and daily consumption
    df['total_qty'] = df['qty_veh_1'] + df['qty_veh_2']
    df['qty_veh_1_daily'] = df['qty_veh_1'] * daily_mult_1
    df['qty_veh_2_daily'] = df['qty_veh_2'] * daily_mult_2
    df['net_daily_consumption'] = df['qty_veh_1_daily'] + df['qty_veh_2_daily']
    
    return df

def merge_vendor_data(master_bom, vendor_df):
    """Merge vendor data with master BOM"""
    vendor_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
    overlap_cols = set(master_bom.columns) & set(vendor_df.columns) - {'part_id'}
    
    if overlap_cols:
        final_df = pd.merge(master_bom, vendor_df, on='part_id', how='left', suffixes=('', '_vendor'))
        for col in overlap_cols:
            vendor_col = f"{col}_vendor"
            if vendor_col in final_df.columns:
                final_df[col] = final_df[col].fillna(final_df[vendor_col])
                final_df.drop(columns=[vendor_col], inplace=True)
    else:
        final_df = pd.merge(master_bom, vendor_df, on='part_id', how='left')
    
    final_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
    return final_df

def show_dashboard():
    st.header("📈 Analysis Dashboard")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ No processed data available. Please upload and process data first.")
        return
    
    df = st.session_state.processed_data
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Parts", len(df))
    
    with col2:
        if 'unit_price' in df.columns:
            total_value = df['unit_price'].sum()
            st.metric("Total Value", f"${total_value:,.2f}")
        else:
            st.metric("Total Value", "N/A")
    
    with col3:
        if 'net_daily_consumption' in df.columns:
            daily_consumption = df['net_daily_consumption'].sum()
            st.metric("Daily Consumption", f"{daily_consumption:,.0f}")
        else:
            st.metric("Daily Consumption", "N/A")
    
    with col4:
        if 'DISTANCE CODE' in df.columns:
            avg_distance_code = df['DISTANCE CODE'].mean()
            st.metric("Avg Distance Code", f"{avg_distance_code:.1f}" if not pd.isna(avg_distance_code) else "N/A")
        else:
            st.metric("Avg Distance Code", "N/A")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Classifications", "📍 Geography", "💰 Financial", "📦 Inventory"])
    
    with tab1:
        show_classification_analysis(df)
    
    with tab2:
        show_geography_analysis(df)
    
    with tab3:
        show_financial_analysis(df)
    
    with tab4:
        show_inventory_analysis(df)

def show_classification_analysis(df):
    st.subheader("Classification Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'family' in df.columns:
            st.markdown("**Family Distribution**")
            family_counts = df['family'].value_counts().head(10)
            fig = px.bar(x=family_counts.values, y=family_counts.index, orientation='h',
                        title="Top 10 Families by Part Count")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'part_classification' in df.columns:
            st.markdown("**Part Classification**")
            part_counts = df['part_classification'].value_counts()
            fig = px.pie(values=part_counts.values, names=part_counts.index,
                        title="Part Classification Distribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Size classification
    if 'size_classification' in df.columns:
        st.markdown("**Size Classification**")
        size_counts = df['size_classification'].value_counts()
        fig = px.bar(x=size_counts.index, y=size_counts.values,
                    title="Size Classification Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_geography_analysis(df):
    st.subheader("Geographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'state' in df.columns:
            st.markdown("**Vendor Distribution by State**")
            state_counts = df['state'].value_counts().head(10)
            fig = px.bar(x=state_counts.values, y=state_counts.index, orientation='h',
                        title="Top 10 States by Vendor Count")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'DISTANCE CODE' in df.columns:
            st.markdown("**Distance Code Distribution**")
            distance_counts = df['DISTANCE CODE'].value_counts().sort_index()
            distance_labels = {1: "< 50 Km", 2: "50-250 Km", 3: "250-750 Km", 4: "> 750 Km"}
            
            labels = [distance_labels.get(i, f"Code {i}") for i in distance_counts.index if not pd.isna(i)]
            values = [distance_counts[i] for i in distance_counts.index if not pd.isna(i)]
            
            fig = px.pie(values=values, names=labels, title="Distance Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Warehouse locations
    if 'wh_loc' in df.columns:
        st.markdown("**Warehouse Location Distribution**")
        wh_counts = df['wh_loc'].value_counts()
        fig = px.treemap(names=wh_counts.index, values=wh_counts.values,
                        title="Warehouse Location Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_financial_analysis(df):
    st.subheader("Financial Analysis")
    
    if 'unit_price' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Distribution by Classification**")
            if 'part_classification' in df.columns:
                fig = px.box(df, x='part_classification', y='unit_price',
                            title="Price Distribution by Part Classification")
                fig.update_yaxis(type="log")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Most Expensive Parts**")
            top_expensive = df.nlargest(10, 'unit_price')[['part_id', 'description', 'unit_price']]
            st.dataframe(top_expensive, use_container_width=True)
        
        # Value analysis
        if 'net_daily_consumption' in df.columns:
            df['daily_value'] = df['unit_price'] * df['net_daily_consumption']
            st.markdown("**Daily Value Consumption**")
            
            daily_value_by_family = df.groupby('family')['daily_value'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=daily_value_by_family.values, y=daily_value_by_family.index, orientation='h',
                        title="Top 10 Families by Daily Value Consumption")
            st.plotly_chart(fig, use_container_width=True)

def show_inventory_analysis(df):
    st.subheader("Inventory Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'RM IN DAYS' in df.columns:
            st.markdown("**Inventory Days Distribution**")
            days_counts = df['RM IN DAYS'].value_counts().sort_index()
            fig = px.bar(x=days_counts.index, y=days_counts.values,
                        title="Raw Material Inventory Days")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'inventory_classification' in df.columns:
            st.markdown("**Inventory Classification**")
            inv_counts = df['inventory_classification'].value_counts()
            fig = px.pie(values=inv_counts.values, names=inv_counts.index,
                        title="Inventory Classification Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Inventory value analysis
    if all(col in df.columns for col in ['RM IN QTY', 'RM IN INR']):
        st.markdown("**Inventory Value Analysis**")
        
        # Top inventory value parts
        top_inventory_value = df.nlargest(15, 'RM IN INR')[
            ['part_id', 'description', 'RM IN QTY', 'RM IN INR', 'RM IN DAYS']
        ]
        st.dataframe(top_inventory_value, use_container_width=True)
        
        # Inventory value by family
        inv_value_by_family = df.groupby('family')['RM IN INR'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=inv_value_by_family.values, y=inv_value_by_family.index, orientation='h',
                    title="Top 10 Families by Inventory Value")
        st.plotly_chart(fig, use_container_width=True)

def show_results_page():
    st.header("📋 Results & Export")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ No processed data available. Please upload and process data first.")
        return
    
    df = st.session_state.processed_data
    
    st.subheader("Data Preview")
    
    # Show key columns
    display_columns = [
        'part_id', 'description', 'family', 'part_classification', 
        'size_classification', 'unit_price', 'vendor_name', 'city', 'state'
    ]
    
    available_display_cols = [col for col in display_columns if col in df.columns]
    
    # Add filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'family' in df.columns:
            families = ['All'] + list(df['family'].unique())
            selected_family = st.selectbox("Filter by Family:", families)
        else:
            selected_family = 'All'
    
    with col2:
        if 'part_classification' in df.columns:
            classifications = ['All'] + list(df['part_classification'].unique())
            selected_classification = st.selectbox("Filter by Part Classification:", classifications)
        else:
            selected_classification = 'All'
    
    with col3:
        if 'state' in df.columns:
            states = ['All'] + list(df['state'].dropna().unique())
            selected_state = st.selectbox("Filter by State:", states)
        else:
            selected_state = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_family != 'All':
        filtered_df = filtered_df[filtered_df['family'] == selected_family]
    
    if selected_classification != 'All':
        filtered_df = filtered_df[filtered_df['part_classification'] == selected_classification]
    
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    
    # Display filtered data
    st.dataframe(filtered_df[available_display_cols].head(100), use_container_width=True)
    
    if len(filtered_df) > 100:
        st.info("Showing first 100 records. Use export to get complete data.")
    
    # Export section
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Complete Dataset (CSV)", type="primary"):
            csv = create_final_export_csv(df)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="complete_inventory_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Filtered Data (CSV)"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered CSV",
                data=csv,
                file_name="filtered_inventory_data.csv",
                mime="text/csv"
            )
    
    # Excel export with formatting
    st.markdown("**Advanced Export Options:**")
    if st.button("📈 Generate Formatted Excel Report"):
        excel_data = create_formatted_excel_data(df)
        st.download_button(
            label="Download Formatted Excel Report",
            data=excel_data,
            file_name="structured_inventory_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def create_final_export_csv(df):
    """Create final CSV export with proper column mapping"""
    final_df = df.copy()
    
    # Enhanced mapping
    enhanced_mapping = {
        'qty_veh_1': 'Qty/Veh 1',
        'qty_veh_2': 'Qty/Veh 2',
        'total_qty': 'TOTAL',
        'qty_veh_1_daily': 'Qty/Veh 1_Daily',
        'qty_veh_2_daily': 'Qty/Veh 2_Daily',
        **PFEP_COLUMN_MAP,
        **INTERNAL_TO_PFEP_NEW_COLS,
        'inventory_classification': 'INVENTORY CLASSIFICATION'
    }
    
    # Rename columns
    existing_rename_map = {k: v for k, v in enhanced_mapping.items() if k in final_df.columns}
    final_df.rename(columns=existing_rename_map, inplace=True)
    
    # Add missing template columns
    for col in ALL_TEMPLATE_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = ''
    
    # Reorder columns according to template
    final_df = final_df[ALL_TEMPLATE_COLUMNS]
    final_df['SR.NO'] = range(1, len(final_df) + 1)
    
    return final_df.to_csv(index=False)

def create_formatted_excel_data(df):
    """Create formatted Excel file in memory"""
    output = io.BytesIO()
    
    final_df = df.copy()
    
    # Apply column mapping
    enhanced_mapping = {
        'qty_veh_1': 'Qty/Veh 1',
        'qty_veh_2': 'Qty/Veh 2',
        'total_qty': 'TOTAL',
        'qty_veh_1_daily': 'Qty/Veh 1_Daily',
        'qty_veh_2_daily': 'Qty/Veh 2_Daily',
        **PFEP_COLUMN_MAP,
        **INTERNAL_TO_PFEP_NEW_COLS,
        'inventory_classification': 'INVENTORY CLASSIFICATION'
    }
    
    existing_rename_map = {k: v for k, v in enhanced_mapping.items() if k in final_df.columns}
    final_df.rename(columns=existing_rename_map, inplace=True)
    
    # Add missing columns
    for col in ALL_TEMPLATE_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = ''
    
    final_df = final_df[ALL_TEMPLATE_COLUMNS]
    final_df['SR.NO'] = range(1, len(final_df) + 1)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Master Data Sheet', index=False)
    
    output.seek(0)
    return output.getvalue()

if __name__ == "__main__":
    main()
