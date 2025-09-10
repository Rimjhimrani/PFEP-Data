import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io
from typing import Optional, Tuple, List, Dict, Any

# Configure Streamlit page
st.set_page_config(
    page_title="Inventory & Supply Chain Analysis System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'master_df' not in st.session_state:
    st.session_state.master_df = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = []

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

WAREHOUSE_LOCATION_FULL_FORMS = {
    "CRL": "CAROUSAL", "HRR": "HIGH RACK", "MEZ B-01": "MEZANNAINE B-01", "MEZ B-01(A)": "MEZANNAINE B-01(A)", "MEZ B-02": "MEZANNAINE B-02",
    "MEZ B-02(A)": "MEZANNAINE B-02(A)", "MEZ C-02": "MEZANNAINE C-02", "MEZ C-02(B)": "MEZANNAINE C-02(B)", "MRR": "MID RISE RACK",
    "MRR(C-01)": "MID RISE RACK (C-01)", "CTR": "CANTILEVER RACK (FASCIA)", "CTR(C-01)": "CANTILEVER RACK (C-01)", "FLOOR": "FLOOR", "OUTSIDE": "OUTSIDE",
    "DIRECT FROM INSTOR": "DIRECT FROM INSTOR"
}

# --- DISTANCE CALCULATION COMPONENTS ---
GEOCODING_CACHE = {}
GEOLOCATOR = Nominatim(user_agent="inventory_distance_calculator_v3", timeout=10)

def get_lat_lon(pincode, country="India", city="", state="", retries=3, backoff_factor=2):
    """Geocodes a location using its pincode, country, city, and state."""
    pincode_str = str(pincode).strip().split('.')[0]
    if not pincode_str.isdigit() or int(pincode_str) == 0:
        return (None, None)

    query_key = f"{pincode_str}|{country}"

    if query_key in GEOCODING_CACHE:
        return GEOCODING_CACHE[query_key]

    if city and state:
        query = f"{pincode_str}, {city}, {state}, {country}"
    else:
        query = f"{pincode_str}, {country}"

    for attempt in range(retries):
        try:
            st.write(f"🔎 Querying for: '{query}' (Attempt {attempt + 1}/{retries})")
            time.sleep(1)

            location = GEOLOCATOR.geocode(query)

            if location:
                coords = (location.latitude, location.longitude)
                GEOCODING_CACHE[query_key] = coords
                return coords

        except Exception as e:
            st.write(f"❌ An exception occurred while geocoding '{pincode_str}': {e}")
            if attempt < retries - 1:
                wait_time = backoff_factor * (attempt + 1)
                st.write(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            continue

    st.write(f"❌ All {retries} attempts failed for pincode: {pincode_str}. It might be invalid or the service is unavailable.")
    GEOCODING_CACHE[query_key] = (None, None)
    return (None, None)

def get_distance_code(distance):
    """Classifies a distance in kilometers into a predefined code."""
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

# --- FILE UPLOAD AND DATA PROCESSING FUNCTIONS ---

def find_qty_veh_column(df):
    """Find the qty/veh column in the DataFrame with various possible names."""
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
    """Automatically finds and renames columns based on the master PFEP_COLUMN_MAP."""
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
    st.write(f"Found and mapped columns: {found_keys}")
    return df, 'part_id' in found_keys

def process_and_diagnose_qty_columns(df):
    """Cleans and diagnoses the Qty/Veh columns."""
    st.write("\n" + "="*50)
    st.write("🩺 Diagnosing Qty/Veh Columns...")
    st.write("="*50)

    # Process Qty/Veh 1
    if 'qty_veh_1' not in df.columns:
        st.write("INFO: 'qty_veh_1' column not found. Creating it with all zeros.")
        df['qty_veh_1'] = 0
    else:
        st.write("\n--- Analysis for 'Qty/Veh 1' ---")
        st.write("Value breakdown before cleaning (NaN represents blank cells):")
        st.write(df['qty_veh_1'].value_counts(dropna=False))
        numeric_col = pd.to_numeric(df['qty_veh_1'], errors='coerce')
        invalid_mask = numeric_col.isna()
        if invalid_mask.sum() > 0:
            st.write(f"\n⚠️ WARNING: Found {invalid_mask.sum()} rows in 'qty_veh_1' that were blank or non-numeric. These have been set to 0.")
        df['qty_veh_1'] = numeric_col.fillna(0)

    # Process Qty/Veh 2
    if 'qty_veh_2' not in df.columns:
        st.write("\nINFO: 'qty_veh_2' column not found. Creating it with all zeros.")
        df['qty_veh_2'] = 0
    else:
        st.write("\n--- Analysis for 'Qty/Veh 2' ---")
        st.write("Value breakdown before cleaning (NaN represents blank cells):")
        st.write(df['qty_veh_2'].value_counts(dropna=False))
        numeric_col = pd.to_numeric(df['qty_veh_2'], errors='coerce')
        invalid_mask = numeric_col.isna()
        if invalid_mask.sum() > 0:
            st.write(f"\n⚠️ WARNING: Found {invalid_mask.sum()} rows in 'qty_veh_2' that were blank or non-numeric. These have been set to 0.")
        df['qty_veh_2'] = numeric_col.fillna(0)

    st.write("\n" + "="*50)
    st.write("✅ Diagnosis complete. All non-numeric quantities set to 0.")
    st.write("="*50)
    return df

# --- PERCENTAGE-BASED PART CLASSIFICATION SYSTEM ---

class PartClassificationSystem:
    def __init__(self):
        self.percentages = {'C': {'target': 60, 'tolerance': 5}, 'B': {'target': 25, 'tolerance': 2}, 'A': {'target': 12, 'tolerance': 2}, 'AA': {'target': 3, 'tolerance': 1}}
        self.calculated_ranges = {}
        self.parts_data = None
        self.manual_classifications = {}

    def load_data_from_dataframe(self, df, price_column='unit_price', part_id_column='part_id'):
        self.parts_data = df.copy()
        if price_column not in self.parts_data.columns:
            raise ValueError(f"Column '{price_column}' not found")
        if part_id_column not in self.parts_data.columns:
            raise ValueError(f"Column '{part_id_column}' not found")
        self.price_column = price_column
        self.part_id_column = part_id_column
        self.calculate_percentage_ranges()
        self.display_calculated_ranges()

    def calculate_percentage_ranges(self):
        if self.parts_data is None:
            return
        valid_prices = [float(p) for p in self.parts_data[self.price_column] if not self.is_blank_price(p)]
        if not valid_prices:
            return
        valid_prices = sorted(valid_prices)
        total_valid_parts = len(valid_prices)
        st.write(f"\nCalculating ranges from {total_valid_parts} valid prices...")
        ranges, cumulative_percent = {}, 0
        for class_name in ['C', 'B', 'A', 'AA']:
            target_percent = self.percentages[class_name]['target']
            start_idx = int((cumulative_percent / 100) * total_valid_parts)
            end_idx = int(((cumulative_percent + target_percent) / 100) * total_valid_parts) - 1
            start_idx, end_idx = max(0, start_idx), min(end_idx, total_valid_parts - 1)
            if start_idx <= end_idx:
                ranges[class_name] = {
                    'min': valid_prices[start_idx], 'max': valid_prices[end_idx], 'actual_count': end_idx - start_idx + 1,
                    'actual_percent': round((end_idx - start_idx + 1) / total_valid_parts * 100, 2),
                    'target_percent': target_percent, 'tolerance': f"±{self.percentages[class_name]['tolerance']}%"
                }
            else:
                ranges[class_name] = {
                    'min': None, 'max': None, 'actual_count': 0, 'actual_percent': 0,
                    'target_percent': target_percent, 'tolerance': f"±{self.percentages[class_name]['tolerance']}%"
                }
            cumulative_percent += target_percent
        self.calculated_ranges = ranges

    def display_calculated_ranges(self):
        if not self.calculated_ranges:
            return
        st.write("\n" + "="*80 + "\nCALCULATED PERCENTAGE-BASED CLASSIFICATION RANGES\n" + "="*80)
        for class_name in ['AA', 'A', 'B', 'C']:
            range_info = self.calculated_ranges.get(class_name)
            if range_info:
                if range_info['min'] is not None:
                    st.write(f"\n{class_name} Class:")
                    st.write(f"  Target: {range_info['target_percent']}% {range_info['tolerance']}")
                    st.write(f"  Actual: {range_info['actual_percent']}% ({range_info['actual_count']} parts)")
                    st.write(f"  Price Range: ${range_info['min']:,.2f} to ${range_info['max']:,.2f}")
                else:
                    st.write(f"\n{class_name} Class:")
                    st.write("  Price Range: No parts in this range")
        st.write("\n" + "="*80)

    def is_blank_price(self, unit_price):
        return pd.isna(unit_price) or str(unit_price).strip().lower() in ['', 'nan', 'null', 'none', 'n/a']

    def classify_part(self, unit_price, part_id=None):
        if self.is_blank_price(unit_price):
            return self.manual_classifications.get(part_id, 'Manual')
        try:
            unit_price = float(unit_price)
        except (ValueError, TypeError):
            return self.manual_classifications.get(part_id, 'Manual')
        for class_name in ['AA', 'A', 'B', 'C']:
            range_info = self.calculated_ranges.get(class_name)
            if range_info and range_info['min'] is not None and range_info['min'] <= unit_price <= range_info['max']:
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

# --- DATA PROCESSING CLASS ---

class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data
        self.rm_days_mapping = {'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11, 'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16, 'C1': 16, 'C2': 31}
        self.classifier = PartClassificationSystem()

    def run_family_classification(self):
        st.write("\n--- (Step 1/6) Running Family Classification ---")
        if 'description' not in self.data.columns:
            return

        def find_kw_pos(desc, kw):
            match = re.search(r'\b' + re.escape(str(kw).upper()) + r'\b', str(desc).upper())
            return match.start() if match else -1

        def extract_family(desc):
            if pd.isna(desc):
                return 'Others'
            for fam in CATEGORY_PRIORITY_FAMILIES:
                if fam in FAMILY_KEYWORD_MAPPING and any(find_kw_pos(desc, kw) != -1 for kw in FAMILY_KEYWORD_MAPPING[fam]):
                    return fam
            matches = [(pos, fam) for fam, kws in FAMILY_KEYWORD_MAPPING.items() 
                      if fam not in CATEGORY_PRIORITY_FAMILIES 
                      for kw in kws for pos in [find_kw_pos(desc, kw)] if pos != -1]
            return min(matches, key=lambda x: x[0])[1] if matches else 'Others'

        self.data['family'] = self.data['description'].apply(extract_family)
        st.write("✅ Automated family classification complete.")

    def run_size_classification(self):
        st.write("\n--- (Step 2/6) Running Size Classification ---")
        if not all(k in self.data.columns for k in ['length', 'width', 'height']):
            return
        for key in ['length', 'width', 'height']:
            self.data[key] = pd.to_numeric(self.data[key], errors='coerce')
        self.data['volume_m3'] = self.data.apply(
            lambda r: (r['length']/1000 * r['width']/1000 * r['height']/1000) 
            if pd.notna(r['length']) and pd.notna(r['width']) and pd.notna(r['height']) else None, axis=1
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
            if (0.5 < row['volume_m3'] <= 1.5) or (750 < max_dim <= 1200):
                return 'L'
            if (0.05 < row['volume_m3'] <= 0.5) or (150 < max_dim <= 750):
                return 'M'
            return 'S'

        self.data['size_classification'] = self.data.apply(classify_size, axis=1)
        st.write("✅ Automated size classification complete.")

    def run_part_classification(self):
        st.write("\n--- (Step 3/6) Running Part Classification (Percentage-Based) ---")
        if 'unit_price' not in self.data.columns or 'part_id' not in self.data.columns:
            self.data['part_classification'] = 'Manual'
            return
        self.classifier.load_data_from_dataframe(self.data)
        classified_df = self.classifier.classify_all_parts()
        if classified_df is not None:
            self.data['part_classification'] = self.data['part_id'].map(
                classified_df.set_index('part_id')['classification'].to_dict()
            )
            st.write("✅ Percentage-based part classification complete.")
        else:
            self.data['part_classification'] = 'Manual'

    def calculate_distances_for_location(self, current_pincode):
        st.write(f"\n⏳ Getting coordinates for current location (pincode: {current_pincode})...")
        current_coords = get_lat_lon(current_pincode, country="India")
        if current_coords == (None, None):
            st.write(f"❌ CRITICAL: Could not find coordinates for your pincode {current_pincode}. Distances cannot be calculated.")
            return [None] * len(self.data)
        st.write(f"✅ Current location found at: {current_coords}")
        st.write(f"\n⏳ Processing vendor locations...")
        
        for col in ['pincode', 'city', 'state']:
            if col not in self.data.columns:
                self.data[col] = ''
        
        distances, distance_codes, failed_locations = [], [], 0
        total_vendors = len(self.data)
        progress_bar = st.progress(0)
        
        for idx, row in self.data.iterrows():
            vendor_pincode = row.get('pincode', '')
            pincode_str = str(vendor_pincode).strip().split('.')[0]
            if not pincode_str or pincode_str.lower() in ['nan', 'none', 'na', 'n/a', '0']:
                distances.append(None)
                distance_codes.append(None)
                continue
            
            vendor_coords = get_lat_lon(
                pincode_str, country="India", 
                city=str(row.get('city', '')).strip(), 
                state=str(row.get('state', '')).strip()
            )
            if vendor_coords == (None, None):
                distances.append(None)
                distance_codes.append(None)
                failed_locations += 1
            else:
                try:
                    distance_km = geodesic(current_coords, vendor_coords).km
                    distances.append(distance_km)
                    distance_codes.append(get_distance_code(distance_km))
                except Exception as e:
                    st.write(f"❌ Error calculating distance for {pincode_str}: {e}")
                    distances.append(None)
                    distance_codes.append(None)
                    failed_locations += 1
            
            # Update progress bar
            progress_bar.progress((idx + 1) / total_vendors)
            if (idx + 1) % 50 == 0:
                st.write(f"   ...processed {idx + 1}/{total_vendors} vendor locations...")
        
        st.write(f"\n📊 Distance Calculation Summary:")
        st.write(f"   - Total vendor records processed: {total_vendors}")
        st.write(f"   - Successfully calculated: {total_vendors - failed_locations - distance_codes.count(None)}")
        st.write(f"   - Failed to geocode (invalid/not found): {failed_locations}")
        st.write(f"   - Skipped (missing pincode): {distance_codes.count(None)}")
        
        if valid_codes := [c for c in distance_codes if c is not None]:
            st.write(f"\n📏 Distance Code Breakdown:")
            for code, count in pd.Series(valid_codes).value_counts().sort_index().items():
                desc = {1: "Less than 50 Km", 2: "50-250 Km", 3: "250-750 Km", 4: "More than 750 Km"}.get(int(code))
                st.write(f"   Code {int(code)} ({desc}): {count} vendors")
        
        return distance_codes

    def run_location_based_norms(self, location_name, pincode):
        st.write(f"\n--- (Step 4 for {location_name}) Running Distance & Inventory Norms ---")
        distance_codes = self.calculate_distances_for_location(pincode)
        distance_code_col = 'DISTANCE CODE'
        self.data[distance_code_col] = distance_codes
        st.write(f"✅ Distance codes calculated for {location_name}.")
        
        if 'part_classification' not in self.data.columns:
            return

        def get_inv_class(p, d):
            if pd.isna(p) or pd.isna(d):
                return None
            d = int(d)
            if p in ['AA', 'A']:
                return f"A{d}"
            if p == 'B':
                return f"B{d}"
            if p == 'C':
                return 'C1' if d in [1, 2] else 'C2'
            return None

        inv_class_col_internal = 'inventory_classification'
        inv_class_col_pfep = 'INVENTORY CLASSIFICATION'
        self.data[inv_class_col_internal] = self.data.apply(
            lambda r: get_inv_class(r.get('part_classification'), r.get(distance_code_col)), axis=1
        )
        INTERNAL_TO_PFEP_NEW_COLS[inv_class_col_internal] = inv_class_col_pfep
        self.data['RM IN DAYS'] = self.data[inv_class_col_internal].map(self.rm_days_mapping)
        self.data['RM IN QTY'] = self.data['RM IN DAYS'] * pd.to_numeric(self.data.get('net_daily_consumption'), errors='coerce')
        self.data['RM IN INR'] = self.data['RM IN QTY'] * pd.to_numeric(self.data.get('unit_price'), errors='coerce')
        self.data['PACKING FACTOR (PF)'] = pd.to_numeric(self.data.get('packing_factor', 1), errors='coerce').fillna(1)
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        self.data['NO OF SEC. PACK REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(self.data['NO OF SEC. PACK REQD.'] * self.data['PACKING FACTOR (PF)'])
        st.write(f"✅ Inventory norms calculated for {location_name}.")

    def run_warehouse_location_assignment(self):
        st.write("\n--- (Step 5/6) Running Warehouse Location Assignment ---")
        if 'family' not in self.data.columns:
            return

        def get_wh_loc(row):
            fam, desc, vol_m3 = row.get('family', 'Others'), row.get('description', ''), row.get('volume_m3', None)
            word_match = lambda text, word: re.search(r'\b' + re.escape(word) + r'\b', str(text).upper())
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
        st.write("✅ Automated warehouse location assignment complete.")

# --- EXCEL REPORT GENERATION ---

def create_formatted_excel_output(df):
    """Creates the final, structured Excel report from the processed DataFrame."""
    st.write("\n--- (Step 6/6) Generating Formatted Excel Report ---")
    final_df = df.copy().loc[:, ~df.columns.duplicated()]
    enhanced_internal_to_pfep_map = {
        'qty_veh_1': 'Qty/Veh 1', 'qty_veh_2': 'Qty/Veh 2', 'total_qty': 'TOTAL', 
        'qty_veh_1_daily': 'Qty/Veh 1_Daily', 'qty_veh_2_daily': 'Qty/Veh 2_Daily', 
        **PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS, 'inventory_classification': 'INVENTORY CLASSIFICATION'
    }
    existing_rename_map = {k: v for k, v in enhanced_internal_to_pfep_map.items() if k in final_df.columns}
    final_df.rename(columns=existing_rename_map, inplace=True)
    
    if any(final_df.columns.duplicated()):
        final_df.columns = pd.io.common.dedup_names(final_df.columns, is_potential_multiindex=False)
    
    for col in [c for c in ALL_TEMPLATE_COLUMNS if c not in final_df.columns]:
        final_df[col] = ''
    
    final_df = final_df[ALL_TEMPLATE_COLUMNS]
    final_df['SR.NO'] = range(1, len(final_df) + 1)
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        h_gray = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 
            'fg_color': '#D9D9D9', 'border': 1
        })
        s_orange = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 
            'fg_color': '#FDE9D9', 'border': 1
        })
        s_blue = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter', 
            'fg_color': '#DCE6F1', 'border': 1
        })
        
        final_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
        worksheet = writer.sheets['Master Data Sheet']
        
        # Add merged headers
        worksheet.merge_range('A1:H1', 'PART DETAILS', h_gray)
        worksheet.merge_range('I1:L1', 'Daily consumption', s_orange)
        worksheet.merge_range('M1:N1', 'PRICE & CLASSIFICATION', s_orange)
        worksheet.merge_range('O1:S1', 'Size & Classification', s_orange)
        worksheet.merge_range('T1:Z1', 'VENDOR DETAILS', s_blue)
        worksheet.merge_range('AA1:AO1', 'PACKAGING DETAILS', s_orange)
        worksheet.merge_range('AP1:AW1', 'INVENTORY NORM', s_blue)
        worksheet.merge_range('AX1:BC1', 'WH STORAGE', s_orange)
        worksheet.merge_range('BD1:BG1', 'SUPPLY SYSTEM', s_blue)
        worksheet.merge_range('BH1:BV1', 'LINE SIDE STORAGE', h_gray)
        worksheet.write('BW1', ALL_TEMPLATE_COLUMNS[-1], h_gray)
        
        for col_num, value in enumerate(final_df.columns):
            worksheet.write(1, col_num, value, h_gray)
        
        worksheet.set_column('A:A', 6)
        worksheet.set_column('B:C', 22)
        worksheet.set_column('D:BW', 18)
    
    excel_buffer.seek(0)
    
    st.write(f"\n✅ Successfully created formatted Excel file")
    st.write(f"📊 Summary of Processing:")
    st.write(f"   - Total parts processed: {len(final_df)}")
    
    if 'DISTANCE CODE' in final_df.columns and (valid_codes := final_df['DISTANCE CODE'].dropna()).any():
        st.write(f"   - Vendor locations with valid distance codes: {len(valid_codes)}")
    
    return excel_buffer, 'structured_inventory_data_final.xlsx'

# --- STREAMLIT UI ---

def main():
    st.title("🏭 COMPREHENSIVE INVENTORY & SUPPLY CHAIN ANALYSIS SYSTEM 🏭")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    if st.session_state.step == 'upload':
        st.sidebar.write("📁 Currently: File Upload")
    elif st.session_state.step == 'daily_consumption':
        st.sidebar.write("📊 Currently: Daily Consumption")
    elif st.session_state.step == 'processing':
        st.sidebar.write("⚙️ Currently: Data Processing")
    elif st.session_state.step == 'location':
        st.sidebar.write("🗺️ Currently: Location Processing")
    elif st.session_state.step == 'complete':
        st.sidebar.write("✅ Currently: Complete")
    
    # Step 1: File Upload
    if st.session_state.step == 'upload':
        st.header("📁 UPLOAD BOM FILES")
        st.markdown("="*50)
        
        # PBOM Files
        st.subheader("📄 Upload PBOM Files")
        pbom_files = st.file_uploader(
            "Upload PBOM files", 
            type=['csv', 'xlsx', 'xls'], 
            accept_multiple_files=True,
            key="pbom_files"
        )
        
        # MBOM Files
        st.subheader("📄 Upload MBOM Files")
        mbom_files = st.file_uploader(
            "Upload MBOM files", 
            type=['csv', 'xlsx', 'xls'], 
            accept_multiple_files=True,
            key="mbom_files"
        )
        
        # Vendor Master File
        st.subheader("🚚 Upload Vendor Master File")
        vendor_file = st.file_uploader(
            "Upload Vendor Master file", 
            type=['csv', 'xlsx', 'xls'],
            key="vendor_file"
        )
        
        if st.button("Process Uploaded Files", type="primary"):
            if not pbom_files and not mbom_files:
                st.error("❌ Please upload at least one BOM file to proceed.")
                return
            
            st.write("Processing uploaded files...")
            all_boms = []
            file_counter = 0
            
            # Process PBOM files
            if pbom_files:
                st.write(f"\n📁 Processing {len(pbom_files)} PBOM files...")
                for uploaded_file in pbom_files:
                    try:
                        if uploaded_file.name.lower().endswith('.csv'):
                            df = pd.read_csv(uploaded_file, low_memory=False)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        file_counter += 1
                        df, has_part_id = find_and_rename_columns(df)
                        
                        if has_part_id:
                            all_boms.append(df)
                            st.success(f"✅ Added {len(df)} records from '{uploaded_file.name}' (File #{file_counter}).")
                        else:
                            st.error(f"❌ Part ID ('{PFEP_COLUMN_MAP['part_id']}') is essential. Skipping file '{uploaded_file.name}'.")
                            file_counter -= 1
                    except Exception as e:
                        st.error(f"❌ Error processing file '{uploaded_file.name}': {e}")
            
            # Process MBOM files
            if mbom_files:
                st.write(f"\n📁 Processing {len(mbom_files)} MBOM files...")
                for i, uploaded_file in enumerate(mbom_files, 1):
                    try:
                        if uploaded_file.name.lower().endswith('.csv'):
                            df = pd.read_csv(uploaded_file, low_memory=False)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        file_counter += 1
                        df, has_part_id = find_and_rename_columns(df, i)
                        
                        if has_part_id:
                            all_boms.append(df)
                            st.success(f"✅ Added {len(df)} records from '{uploaded_file.name}' (File #{file_counter}).")
                        else:
                            st.error(f"❌ Part ID ('{PFEP_COLUMN_MAP['part_id']}') is essential. Skipping file '{uploaded_file.name}'.")
                            file_counter -= 1
                    except Exception as e:
                        st.error(f"❌ Error processing file '{uploaded_file.name}': {e}")
            
            if not all_boms:
                st.error("❌ No valid BOM data loaded. Cannot proceed.")
                return
            
            # Consolidate BOM data
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
            
            # Process qty columns
            master_bom = process_and_diagnose_qty_columns(master_bom)
            master_bom['total_qty'] = master_bom['qty_veh_1'] + master_bom['qty_veh_2']
            
            # Process vendor file
            if vendor_file:
                st.write("\n🚚 Processing Vendor Master file...")
                try:
                    if vendor_file.name.lower().endswith('.csv'):
                        vendor_df = pd.read_csv(vendor_file, low_memory=False)
                    else:
                        vendor_df = pd.read_excel(vendor_file)
                    
                    vendor_df, has_part_id = find_and_rename_columns(vendor_df)
                    if has_part_id:
                        vendor_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
                        overlap_cols = set(master_bom.columns) & set(vendor_df.columns) - {'part_id'}
                        if overlap_cols:
                            master_bom = pd.merge(master_bom, vendor_df, on='part_id', how='left', suffixes=('', '_vendor'))
                            for col in overlap_cols:
                                vendor_col = f"{col}_vendor"
                                if vendor_col in master_bom.columns:
                                    master_bom[col] = master_bom[col].fillna(master_bom[vendor_col])
                                    master_bom.drop(columns=[vendor_col], inplace=True)
                        else:
                            master_bom = pd.merge(master_bom, vendor_df, on='part_id', how='left')
                        st.success("✅ Vendor data successfully merged.")
                    else:
                        st.warning("⚠️ Part ID not found in vendor file. Cannot merge.")
                except Exception as e:
                    st.error(f"❌ Error processing vendor file: {e}")
            
            master_bom.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
            st.session_state.master_df = master_bom
            st.session_state.step = 'daily_consumption'
            st.rerun()
    
    # Step 2: Daily Consumption Configuration
    elif st.session_state.step == 'daily_consumption':
        st.header("📊 DAILY CONSUMPTION CALCULATION")
        st.markdown("="*50)
        
        st.write(f"Total unique parts loaded: {len(st.session_state.master_df)}")
        
        col1, col2 = st.columns(2)
        with col1:
            qty_veh_1_daily = st.number_input(
                "Enter daily production quantity for Vehicle Type 1:",
                min_value=0.0, value=1.0, step=1.0
            )
        with col2:
            qty_veh_2_daily = st.number_input(
                "Enter daily production quantity for Vehicle Type 2:",
                min_value=0.0, value=1.0, step=1.0
            )
        
        if st.button("Calculate Daily Consumption", type="primary"):
            master_df = st.session_state.master_df
            master_df['qty_veh_1_daily'] = master_df['qty_veh_1'] * qty_veh_1_daily
            master_df['qty_veh_2_daily'] = master_df['qty_veh_2'] * qty_veh_2_daily
            master_df['net_daily_consumption'] = master_df['qty_veh_1_daily'] + master_df['qty_veh_2_daily']
            
            st.session_state.master_df = master_df
            st.success(f"✅ Daily consumption calculations complete. Total unique parts: {len(master_df)}")
            
            # Initialize processor
            processor = ComprehensiveInventoryProcessor(master_df.loc[:, ~master_df.columns.duplicated()])
            st.session_state.processor = processor
            st.session_state.step = 'processing'
            st.rerun()
    
    # Step 3: Data Processing
    elif st.session_state.step == 'processing':
        st.header("⚙️ DATA PROCESSING STEPS")
        st.markdown("="*50)
        
        processor = st.session_state.processor
        
        # Family Classification
        if st.button("Run Family Classification (Step 1/6)", type="secondary"):
            with st.spinner("Running family classification..."):
                processor.run_family_classification()
            st.success("Family classification completed!")
        
        # Size Classification
        if st.button("Run Size Classification (Step 2/6)", type="secondary"):
            with st.spinner("Running size classification..."):
                processor.run_size_classification()
            st.success("Size classification completed!")
        
        # Part Classification
        if st.button("Run Part Classification (Step 3/6)", type="secondary"):
            with st.spinner("Running part classification..."):
                processor.run_part_classification()
            st.success("Part classification completed!")
        
        if st.button("Proceed to Location Processing", type="primary"):
            st.session_state.step = 'location'
            st.rerun()
    
    # Step 4: Location Processing
    elif st.session_state.step == 'location':
        st.header("🗺️ LOCATION-BASED PROCESSING")
        st.markdown("="*50)
        
        current_pincode = st.text_input(
            "Enter your current pincode for distance calculation:",
            value="411001",
            placeholder="e.g., 411001"
        )
        
        if st.button("Run Location-Based Processing", type="primary"):
            if not current_pincode.strip():
                st.error("Please enter a valid pincode.")
                return
            
            processor = st.session_state.processor
            
            with st.spinner("Processing location-based norms..."):
                processor.run_location_based_norms("Pune", current_pincode.strip())
            
            with st.spinner("Running warehouse location assignment..."):
                processor.run_warehouse_location_assignment()
            
            st.success("Location processing completed!")
            
            # Generate final report
            with st.spinner("Generating formatted Excel report..."):
                excel_buffer, filename = create_formatted_excel_output(processor.data)
            
            st.session_state.excel_buffer = excel_buffer
            st.session_state.filename = filename
            st.session_state.step = 'complete'
            st.rerun()
    
    # Step 5: Complete
    elif st.session_state.step == 'complete':
        st.header("✅ PROCESS COMPLETE!")
        st.markdown("="*50)
        
        st.success("🎉 End-to-end process complete!")
        
        # Download button
        st.download_button(
            label="📥 Download Final Excel Report",
            data=st.session_state.excel_buffer,
            file_name=st.session_state.filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
        
        # Show summary
        if st.session_state.processor:
            final_data = st.session_state.processor.data
            st.write("📊 **Processing Summary:**")
            st.write(f"- Total parts processed: {len(final_data)}")
            
            if 'DISTANCE CODE' in final_data.columns:
                valid_codes = final_data['DISTANCE CODE'].dropna()
                if len(valid_codes) > 0:
                    st.write(f"- Vendor locations with valid distance codes: {len(valid_codes)}")
            
            # Show data preview
            with st.expander("📋 Preview Final Data"):
                st.dataframe(final_data.head(20))
        
        if st.button("🔄 Start New Analysis", type="secondary"):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.step = 'upload'
            st.session_state.master_df = None
            st.session_state.processor = None
            st.session_state.uploaded_files_data = []
            st.rerun()

if __name__ == "__main__":
    main()
