import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io
import base64
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Inventory & Supply Chain Analysis System",
    page_icon="🏭",
    layout="wide"
)

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
    'family': 'FAMILY', 
    'part_classification': 'PART CLASSIFICATION', 
    'volume_m3': 'Volume (m^3)',
    'size_classification': 'SIZE CLASSIFICATION', 
    'wh_loc': 'WH LOC'
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

CATEGORY_PRIORITY_FAMILIES = {
    "ACP Sheet", "ADAPTOR", "Bracket", "Bush", "Flap", "Handle", 
    "Beading", "Lubricants", "Panel", "Pillar", "Rail", "Seal", "Sticker", "Valve"
}

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
if 'geocoding_cache' not in st.session_state:
    st.session_state.geocoding_cache = {}

@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="inventory_distance_calculator_streamlit", timeout=10)

def get_lat_lon(pincode, country="India", city="", state="", retries=3, backoff_factor=2):
    """Geocodes a location using its pincode, country, city, and state."""
    geolocator = get_geolocator()
    
    pincode_str = str(pincode).strip().split('.')[0]
    if not pincode_str.isdigit() or int(pincode_str) == 0:
        return (None, None)

    query_key = f"{pincode_str}|{country}"
    
    if query_key in st.session_state.geocoding_cache:
        return st.session_state.geocoding_cache[query_key]

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
                wait_time = backoff_factor * (attempt + 1)
                time.sleep(wait_time)
            continue

    st.session_state.geocoding_cache[query_key] = (None, None)
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

# --- DATA LOADING AND PROCESSING FUNCTIONS ---

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
    return df, 'part_id' in found_keys

def process_and_diagnose_qty_columns(df):
    """Cleans and diagnoses the Qty/Veh columns."""
    
    if 'qty_veh_1' not in df.columns:
        df['qty_veh_1'] = 0
    else:
        numeric_col = pd.to_numeric(df['qty_veh_1'], errors='coerce')
        invalid_mask = numeric_col.isna()
        if invalid_mask.sum() > 0:
            st.warning(f"Found {invalid_mask.sum()} rows in 'qty_veh_1' that were blank or non-numeric. These have been set to 0.")
        df['qty_veh_1'] = numeric_col.fillna(0)

    if 'qty_veh_2' not in df.columns:
        df['qty_veh_2'] = 0
    else:
        numeric_col = pd.to_numeric(df['qty_veh_2'], errors='coerce')
        invalid_mask = numeric_col.isna()
        if invalid_mask.sum() > 0:
            st.warning(f"Found {invalid_mask.sum()} rows in 'qty_veh_2' that were blank or non-numeric. These have been set to 0.")
        df['qty_veh_2'] = numeric_col.fillna(0)

    return df

# --- PART CLASSIFICATION SYSTEM ---

class PartClassificationSystem:
    def __init__(self):
        self.percentages = {
            'C': {'target': 60, 'tolerance': 5}, 
            'B': {'target': 25, 'tolerance': 2}, 
            'A': {'target': 12, 'tolerance': 2}, 
            'AA': {'target': 3, 'tolerance': 1}
        }
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
        
        ranges, cumulative_percent = {}, 0
        for class_name in ['C', 'B', 'A', 'AA']:
            target_percent = self.percentages[class_name]['target']
            start_idx = int((cumulative_percent / 100) * total_valid_parts)
            end_idx = int(((cumulative_percent + target_percent) / 100) * total_valid_parts) - 1
            start_idx, end_idx = max(0, start_idx), min(end_idx, total_valid_parts - 1)
            
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
                    'min': None, 'max': None, 'actual_count': 0, 'actual_percent': 0, 
                    'target_percent': target_percent, 'tolerance': f"±{self.percentages[class_name]['tolerance']}%"
                }
            cumulative_percent += target_percent
        
        self.calculated_ranges = ranges

    def display_calculated_ranges(self):
        if not self.calculated_ranges:
            return
        
        st.subheader("📊 Calculated Percentage-Based Classification Ranges")
        
        for class_name in ['AA', 'A', 'B', 'C']:
            range_info = self.calculated_ranges.get(class_name)
            if range_info:
                if range_info['min'] is not None:
                    st.write(f"**{class_name} Class:**")
                    st.write(f"  - Target: {range_info['target_percent']}% {range_info['tolerance']}")
                    st.write(f"  - Actual: {range_info['actual_percent']}% ({range_info['actual_count']} parts)")
                    st.write(f"  - Price Range: ${range_info['min']:,.2f} to ${range_info['max']:,.2f}")
                else:
                    st.write(f"**{class_name} Class:** No parts in this range")

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

# --- COMPREHENSIVE INVENTORY PROCESSOR ---

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
        st.subheader("🏷️ Step 1/6: Family Classification")
        
        if 'description' not in self.data.columns:
            st.warning("Description column not found. Skipping family classification.")
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
            
            # Non-priority families
            matches = []
            for fam, kws in FAMILY_KEYWORD_MAPPING.items():
                if fam not in CATEGORY_PRIORITY_FAMILIES:
                    for kw in kws:
                        pos = find_kw_pos(desc, kw)
                        if pos != -1:
                            matches.append((pos, fam))
            
            return min(matches, key=lambda x: x[0])[1] if matches else 'Others'

        self.data['family'] = self.data['description'].apply(extract_family)
        
        # Display family classification results
        family_counts = self.data['family'].value_counts()
        st.success(f"✅ Automated family classification complete. Found {len(family_counts)} different families.")
        
        with st.expander("View Family Distribution"):
            st.write(family_counts)

    def run_size_classification(self):
        st.subheader("📏 Step 2/6: Size Classification")
        
        required_cols = ['length', 'width', 'height']
        if not all(k in self.data.columns for k in required_cols):
            st.warning("Required dimension columns not found. Skipping size classification.")
            return

        # Convert to numeric
        for key in required_cols:
            self.data[key] = pd.to_numeric(self.data[key], errors='coerce')

        # Calculate volume
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
        
        # Display size classification results
        size_counts = self.data['size_classification'].value_counts()
        st.success(f"✅ Automated size classification complete.")
        
        with st.expander("View Size Classification Distribution"):
            st.write(size_counts)

    def run_part_classification(self):
        st.subheader("💰 Step 3/6: Part Classification (Percentage-Based)")
        
        if 'unit_price' not in self.data.columns or 'part_id' not in self.data.columns:
            self.data['part_classification'] = 'Manual'
            st.warning("Required columns not found. Setting all classifications to 'Manual'.")
            return

        try:
            self.classifier.load_data_from_dataframe(self.data)
            classified_df = self.classifier.classify_all_parts()
            
            if classified_df is not None:
                self.data['part_classification'] = self.data['part_id'].map(
                    classified_df.set_index('part_id')['classification'].to_dict()
                )
                st.success("✅ Percentage-based part classification complete.")
                
                # Display classification results
                classification_counts = self.data['part_classification'].value_counts()
                with st.expander("View Part Classification Distribution"):
                    st.write(classification_counts)
            else:
                self.data['part_classification'] = 'Manual'
                st.warning("Classification failed. Setting all to 'Manual'.")
                
        except Exception as e:
            st.error(f"Error in part classification: {e}")
            self.data['part_classification'] = 'Manual'

    def calculate_distances_for_location(self, current_pincode):
        st.write(f"⏳ Getting coordinates for current location (pincode: {current_pincode})...")
        
        current_coords = get_lat_lon(current_pincode, country="India")
        if current_coords == (None, None):
            st.error(f"❌ CRITICAL: Could not find coordinates for your pincode {current_pincode}. Distances cannot be calculated.")
            return [None] * len(self.data)
        
        st.success(f"✅ Current location found at: {current_coords}")
        
        # Prepare vendor location columns
        for col in ['pincode', 'city', 'state']:
            if col not in self.data.columns:
                self.data[col] = ''

        distances, distance_codes, failed_locations = [], [], 0
        total_vendors = len(self.data)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in self.data.iterrows():
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
                failed_locations += 1
            else:
                try:
                    distance_km = geodesic(current_coords, vendor_coords).km
                    distances.append(distance_km)
                    distance_codes.append(get_distance_code(distance_km))
                except Exception as e:
                    distances.append(None)
                    distance_codes.append(None)
                    failed_locations += 1

            # Update progress
            progress = (idx + 1) / total_vendors
            progress_bar.progress(progress)
            status_text.text(f"Processing vendor locations... {idx + 1}/{total_vendors}")

        progress_bar.empty()
        status_text.empty()
        
        # Display summary
        st.write("📊 **Distance Calculation Summary:**")
        st.write(f"- Total vendor records processed: {total_vendors}")
        st.write(f"- Successfully calculated distances: {len([d for d in distances if d is not None])}")
        st.write(f"- Failed geocoding attempts: {failed_locations}")
        
        if failed_locations > 0:
            st.warning(f"⚠️ {failed_locations} vendor locations could not be geocoded. These will show as 'None' in distance calculations.")
        
        return distances, distance_codes

    def run_distance_classification(self):
        st.subheader("🗺️ Step 4/6: Distance Classification")
        
        # Get current location from user
        current_pincode = st.text_input(
            "Enter your current location pincode:", 
            value="411014",
            help="This will be used to calculate distances to all vendors"
        )
        
        if not current_pincode:
            st.warning("Please enter a pincode to calculate distances.")
            return
        
        try:
            distances, distance_codes = self.calculate_distances_for_location(current_pincode)
            self.data['distance_km'] = distances
            self.data['distance_code'] = distance_codes
            
            st.success("✅ Distance classification complete.")
            
            # Display distance code distribution
            distance_code_counts = pd.Series(distance_codes).value_counts().sort_index()
            with st.expander("View Distance Code Distribution"):
                st.write("Distance Codes:")
                st.write("- Code 1: < 50 km")
                st.write("- Code 2: 50-250 km") 
                st.write("- Code 3: 250-750 km")
                st.write("- Code 4: > 750 km")
                st.write("\nActual Distribution:")
                st.write(distance_code_counts)
                
        except Exception as e:
            st.error(f"Error in distance calculation: {e}")
            self.data['distance_km'] = None
            self.data['distance_code'] = None

    def run_inventory_classification(self):
        st.subheader("📦 Step 5/6: Inventory Classification")
        
        required_cols = ['part_classification', 'distance_code']
        if not all(col in self.data.columns for col in required_cols):
            st.warning("Required columns for inventory classification not found.")
            self.data['inventory_classification'] = 'Manual'
            return
        
        def get_inventory_class(part_class, dist_code):
            if pd.isna(part_class) or pd.isna(dist_code) or part_class == 'Manual':
                return 'Manual'
            
            if part_class in ['AA', 'A']:
                return f'A{dist_code}'
            elif part_class == 'B':
                return f'B{dist_code}' if dist_code in [1, 2] else 'C1'
            elif part_class == 'C':
                return 'C1' if dist_code in [1, 2] else 'C2'
            else:
                return 'Manual'
        
        self.data['inventory_classification'] = self.data.apply(
            lambda row: get_inventory_class(row['part_classification'], row['distance_code']), 
            axis=1
        )
        
        st.success("✅ Inventory classification complete.")
        
        # Display inventory classification results
        inv_counts = self.data['inventory_classification'].value_counts()
        with st.expander("View Inventory Classification Distribution"):
            st.write(inv_counts)

    def calculate_rm_requirements(self):
        st.subheader("📋 Step 6/6: RM Requirements Calculation")
        
        # Calculate NET (daily consumption)
        if 'qty_veh_1' in self.data.columns and 'qty_veh_2' in self.data.columns:
            self.data['net_daily_consumption'] = self.data['qty_veh_1'] + self.data['qty_veh_2']
        elif 'qty_veh' in self.data.columns:
            self.data['net_daily_consumption'] = self.data['qty_veh']
        else:
            self.data['net_daily_consumption'] = 0
            st.warning("No quantity per vehicle columns found. Setting NET consumption to 0.")
        
        # Calculate RM days, quantity, and value
        def calculate_rm_values(row):
            inv_class = row.get('inventory_classification', 'Manual')
            rm_days = self.rm_days_mapping.get(inv_class, 0)
            
            net_consumption = row.get('net_daily_consumption', 0)
            unit_price = row.get('unit_price', 0)
            
            # Handle non-numeric values
            try:
                net_consumption = float(net_consumption) if not pd.isna(net_consumption) else 0
                unit_price = float(unit_price) if not pd.isna(unit_price) else 0
            except (ValueError, TypeError):
                net_consumption = 0
                unit_price = 0
            
            rm_qty = rm_days * net_consumption
            rm_inr = rm_qty * unit_price
            
            return rm_days, rm_qty, rm_inr
        
        rm_calculations = self.data.apply(calculate_rm_values, axis=1)
        self.data['rm_days'] = [calc[0] for calc in rm_calculations]
        self.data['rm_qty'] = [calc[1] for calc in rm_calculations]
        self.data['rm_inr'] = [calc[2] for calc in rm_calculations]
        
        st.success("✅ RM requirements calculation complete.")
        
        # Display RM summary
        with st.expander("View RM Requirements Summary"):
            total_rm_value = self.data['rm_inr'].sum()
            total_parts = len(self.data)
            st.write(f"- Total parts analyzed: {total_parts}")
            st.write(f"- Total RM value: ₹{total_rm_value:,.2f}")
            st.write(f"- Average RM value per part: ₹{total_rm_value/total_parts:,.2f}" if total_parts > 0 else "- Average RM value per part: ₹0")

    def assign_warehouse_locations(self):
        st.subheader("🏭 Warehouse Location Assignment")
        
        if 'family' not in self.data.columns:
            st.warning("Family classification not found. Cannot assign warehouse locations.")
            self.data['wh_loc'] = 'HRR'  # Default location
            return
        
        self.data['wh_loc'] = self.data['family'].map(BASE_WAREHOUSE_MAPPING).fillna('HRR')
        
        st.success("✅ Warehouse location assignment complete.")
        
        # Display warehouse distribution
        wh_counts = self.data['wh_loc'].value_counts()
        with st.expander("View Warehouse Location Distribution"):
            st.write(wh_counts)

    def process_complete_inventory(self):
        st.header("🔄 Complete Inventory Processing Pipeline")
        
        processing_steps = [
            ("Family Classification", self.run_family_classification),
            ("Size Classification", self.run_size_classification), 
            ("Part Classification", self.run_part_classification),
            ("Distance Classification", self.run_distance_classification),
            ("Inventory Classification", self.run_inventory_classification),
            ("RM Requirements", self.calculate_rm_requirements),
            ("Warehouse Assignment", self.assign_warehouse_locations)
        ]
        
        for step_name, step_function in processing_steps:
            try:
                step_function()
                st.success(f"✅ {step_name} completed successfully")
            except Exception as e:
                st.error(f"❌ Error in {step_name}: {str(e)}")
                continue
        
        st.success("🎉 Complete inventory processing pipeline finished!")
        return self.data

# --- PFEP TEMPLATE GENERATOR ---

def generate_pfep_template(processed_data):
    st.header("📊 PFEP Template Generation")
    
    template_data = pd.DataFrame(columns=ALL_TEMPLATE_COLUMNS)
    
    # Fill template with processed data
    for idx, row in processed_data.iterrows():
        template_row = {}
        
        # Basic information
        template_row['SR.NO'] = idx + 1
        template_row['PARTNO'] = row.get('part_id', '')
        template_row['PART DESCRIPTION'] = row.get('description', '')
        template_row['Qty/Veh 1'] = row.get('qty_veh_1', 0)
        template_row['Qty/Veh 2'] = row.get('qty_veh_2', 0)
        template_row['TOTAL'] = template_row['Qty/Veh 1'] + template_row['Qty/Veh 2']
        template_row['NET'] = row.get('net_daily_consumption', 0)
        template_row['UNIT PRICE'] = row.get('unit_price', 0)
        
        # Classifications
        template_row['FAMILY'] = row.get('family', '')
        template_row['PART CLASSIFICATION'] = row.get('part_classification', '')
        template_row['SIZE CLASSIFICATION'] = row.get('size_classification', '')
        template_row['INVENTORY CLASSIFICATION'] = row.get('inventory_classification', '')
        
        # Dimensions
        template_row['L-MM_Size'] = row.get('length', 0)
        template_row['W-MM_Size'] = row.get('width', 0) 
        template_row['H-MM_Size'] = row.get('height', 0)
        template_row['Volume (m^3)'] = row.get('volume_m3', 0)
        
        # Vendor information
        template_row['VENDOR CODE'] = row.get('vendor_code', '')
        template_row['VENDOR NAME'] = row.get('vendor_name', '')
        template_row['CITY'] = row.get('city', '')
        template_row['STATE'] = row.get('state', '')
        template_row['COUNTRY'] = row.get('country', 'India')
        template_row['PINCODE'] = row.get('pincode', '')
        template_row['DISTANCE CODE'] = row.get('distance_code', '')
        
        # RM calculations
        template_row['RM IN DAYS'] = row.get('rm_days', 0)
        template_row['RM IN QTY'] = row.get('rm_qty', 0)
        template_row['RM IN INR'] = row.get('rm_inr', 0)
        
        # Warehouse location
        template_row['WH LOC'] = row.get('wh_loc', 'HRR')
        
        # Packing information (if available)
        template_row['PACKING FACTOR (PF)'] = row.get('packing_factor', 1)
        template_row['QTY/PACK_Sec'] = row.get('qty_per_pack', 1)
        
        # Add other columns with default values
        for col in ALL_TEMPLATE_COLUMNS:
            if col not in template_row:
                template_row[col] = ''
        
        template_data = pd.concat([template_data, pd.DataFrame([template_row])], ignore_index=True)
    
    return template_data

# --- MAIN STREAMLIT APPLICATION ---

def main():
    st.title("🏭 Comprehensive Inventory & Supply Chain Analysis System")
    st.markdown("### Upload your inventory data and get complete PFEP analysis")
    
    # Sidebar for navigation
    st.sidebar.header("📋 Navigation")
    analysis_mode = st.sidebar.selectbox(
        "Select Analysis Mode:",
        ["Single File Analysis", "Two File Comparison", "Template Generator"]
    )
    
    if analysis_mode == "Single File Analysis":
        st.header("📁 Single File Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload your inventory CSV file", 
            type=['csv'], 
            help="Upload a CSV file containing your inventory data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                with st.expander("View Raw Data Preview"):
                    st.dataframe(df.head())
                
                # Process the data
                df, has_part_id = find_and_rename_columns(df)
                
                if not has_part_id:
                    st.error("❌ Part ID column not found in the uploaded file. Please ensure your file has a part number/ID column.")
                    return
                
                df = process_and_diagnose_qty_columns(df)
                
                # Run comprehensive analysis
                processor = ComprehensiveInventoryProcessor(df)
                processed_data = processor.process_complete_inventory()
                
                # Generate PFEP template
                pfep_template = generate_pfep_template(processed_data)
                
                st.header("📥 Download Results")
                
                # Create download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    processed_csv = processed_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Processed Data",
                        data=processed_csv,
                        file_name="processed_inventory_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    template_csv = pfep_template.to_csv(index=False)
                    st.download_button(
                        label="📋 Download PFEP Template",
                        data=template_csv,
                        file_name="complete_pfep_template.csv",
                        mime="text/csv"
                    )
                
                # Display final summary
                st.header("📈 Final Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Parts", len(processed_data))
                
                with col2:
                    total_rm_value = processed_data['rm_inr'].sum() if 'rm_inr' in processed_data.columns else 0
                    st.metric("Total RM Value", f"₹{total_rm_value:,.0f}")
                
                with col3:
                    families_count = processed_data['family'].nunique() if 'family' in processed_data.columns else 0
                    st.metric("Part Families", families_count)
                
                with col4:
                    vendors_count = processed_data['vendor_code'].nunique() if 'vendor_code' in processed_data.columns else 0
                    st.metric("Unique Vendors", vendors_count)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please check your file format and ensure it contains the required columns.")
    
    elif analysis_mode == "Two File Comparison":
        st.header("📊 Two File Comparison Analysis")
        st.info("Upload two inventory files to compare and merge their data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File 1 (Vehicle Type 1)")
            file1 = st.file_uploader("Upload first inventory file", type=['csv'], key="file1")
        
        with col2:
            st.subheader("File 2 (Vehicle Type 2)")  
            file2 = st.file_uploader("Upload second inventory file", type=['csv'], key="file2")
        
        if file1 is not None and file2 is not None:
            try:
                df1 = pd.read_csv(file1)
                df2 = pd.read_csv(file2)
                
                st.success(f"✅ Files uploaded! File 1: {df1.shape}, File 2: {df2.shape}")
                
                # Process both files
                df1, has_part_id1 = find_and_rename_columns(df1, file_number=1)
                df2, has_part_id2 = find_and_rename_columns(df2, file_number=2)
                
                if not (has_part_id1 and has_part_id2):
                    st.error("❌ Part ID columns not found in one or both files.")
                    return
                
                # Merge the files on part_id
                merged_df = df1.merge(df2, on='part_id', how='outer', suffixes=('_1', '_2'))
                
                # Fill missing columns and process
                merged_df = process_and_diagnose_qty_columns(merged_df)
                
                # Combine other columns intelligently
                for col in ['description', 'unit_price', 'vendor_code', 'vendor_name', 'city', 'state', 'country', 'pincode']:
                    if f'{col}_1' in merged_df.columns and f'{col}_2' in merged_df.columns:
                        merged_df[col] = merged_df[f'{col}_1'].fillna(merged_df[f'{col}_2'])
                    elif f'{col}_1' in merged_df.columns:
                        merged_df[col] = merged_df[f'{col}_1']
                    elif f'{col}_2' in merged_df.columns:
                        merged_df[col] = merged_df[f'{col}_2']
                
                st.success(f"✅ Files merged successfully! Combined shape: {merged_df.shape}")
                
                # Run comprehensive analysis on merged data
                processor = ComprehensiveInventoryProcessor(merged_df)
                processed_data = processor.process_complete_inventory()
                
                # Generate PFEP template
                pfep_template = generate_pfep_template(processed_data)
                
                # Download buttons
                st.header("📥 Download Merged Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    processed_csv = processed_data.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Merged Processed Data",
                        data=processed_csv,
                        file_name="merged_processed_inventory.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    template_csv = pfep_template.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Complete PFEP Template", 
                        data=template_csv,
                        file_name="merged_pfep_template.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
    
    elif analysis_mode == "Template Generator":
        st.header("📋 PFEP Template Generator")
        st.info("Generate a blank PFEP template or upload data to create a populated template")
        
        template_option = st.radio(
            "Choose template option:",
            ["Generate Blank Template", "Upload Data for Populated Template"]
        )
        
        if template_option == "Generate Blank Template":
            blank_template = pd.DataFrame(columns=ALL_TEMPLATE_COLUMNS)
            
            # Add some sample rows
            for i in range(5):
                blank_template = pd.concat([blank_template, pd.DataFrame([{col: '' for col in ALL_TEMPLATE_COLUMNS}])], ignore_index=True)
            
            st.success("✅ Blank PFEP template generated!")
            
            template_csv = blank_template.to_csv(index=False)
            st.download_button(
                label="📋 Download Blank PFEP Template",
                data=template_csv,
                file_name="blank_pfep_template.csv", 
                mime="text/csv"
            )
            
            with st.expander("View Template Structure"):
                st.write("Template contains the following columns:")
                st.write(ALL_TEMPLATE_COLUMNS)
        
        else:
            uploaded_file = st.file_uploader(
                "Upload your data file", 
                type=['csv'],
                help="Upload any CSV file to generate a populated PFEP template"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File uploaded! Processing {len(df)} records...")
                    
                    # Basic processing
                    df, _ = find_and_rename_columns(df)
                    pfep_template = generate_pfep_template(df)
                    
                    st.success("✅ Populated PFEP template generated!")
                    
                    template_csv = pfep_template.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Populated PFEP Template",
                        data=template_csv,
                        file_name="populated_pfep_template.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating template: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏭 Comprehensive Inventory & Supply Chain Analysis System</p>
        <p>Automated PFEP Generation | Distance Calculation | Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
