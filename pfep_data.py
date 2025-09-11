import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io

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

# --- ENHANCED UI FUNCTIONS ---
def display_step_card(step_num, title, description, icon="📊", status="pending"):
    """Display an enhanced step card with status indicators"""
    status_colors = {
        "pending": "#f0f2f6",
        "processing": "#fff3cd", 
        "completed": "#d4edda",
        "error": "#f8d7da"
    }
    
    status_icons = {
        "pending": "⏳",
        "processing": "🔄", 
        "completed": "✅",
        "error": "❌"
    }
    
    bg_color = status_colors.get(status, "#f0f2f6")
    status_icon = status_icons.get(status, "⏳")
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {bg_color} 0%, #ffffff 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.2s;
    ">
        <h3 style="margin: 0; color: #2c3e50; display: flex; align-items: center;">
            {icon} Step {step_num}: {title} {status_icon}
        </h3>
        <p style="margin: 10px 0 0 0; color: #7f8c8d;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None, color="#4CAF50"):
    """Create enhanced metric cards"""
    delta_html = f"<p style='color: {color}; margin: 5px 0 0 0; font-size: 14px;'>{delta}</p>" if delta else ""
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    ">
        <h3 style="color: #2c3e50; margin: 0; font-size: 16px;">{title}</h3>
        <h1 style="color: {color}; margin: 10px 0; font-size: 32px;">{value}</h1>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_progress_indicator(steps_completed, total_steps):
    """Create a visual progress indicator"""
    progress = steps_completed / total_steps
    st.markdown(f"""
    <div style="margin: 20px 0;">
        <h4 style="color: #2c3e50;">Overall Progress: {steps_completed}/{total_steps} Steps Completed</h4>
        <div style="
            background-color: #e9ecef;
            border-radius: 10px;
            height: 10px;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
                height: 100%;
                width: {progress * 100}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <p style="color: #6c757d; font-size: 14px; margin-top: 5px;">
            {progress * 100:.1f}% Complete
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- 6. MANUAL REVIEW STEP WITH ENHANCED UI ---
def manual_review_step(df, internal_key, step_name):
    st.markdown("---")
    st.markdown(f"### 🔍 Manual Review: {step_name}")
    
    pfep_name = INTERNAL_TO_PFEP_NEW_COLS.get(internal_key, PFEP_COLUMN_MAP.get(internal_key, internal_key))
    review_df = df[['part_id', 'description', internal_key]].copy()
    review_df.rename(columns={internal_key: pfep_name, 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION'}, inplace=True)
    
    # Enhanced preview with metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        create_metric_card("Total Parts", len(review_df), "📦")
    with col2:
        unique_values = review_df[pfep_name].nunique()
        create_metric_card("Unique Categories", unique_values, "🏷️")
    with col3:
        null_count = review_df[pfep_name].isnull().sum()
        create_metric_card("Missing Values", null_count, "⚠️", "#ff6b6b" if null_count > 0 else "#4CAF50")
    
    # Enhanced data preview
    st.markdown("#### 📊 Data Preview")
    with st.expander("View Data Sample", expanded=False):
        st.dataframe(
            review_df.head(10), 
            use_container_width=True,
            hide_index=True
        )
    
    # Value distribution
    if pfep_name in review_df.columns:
        st.markdown("#### 📈 Classification Distribution")
        value_counts = review_df[pfep_name].value_counts()
        st.bar_chart(value_counts)
    
    # Download section with enhanced styling
    st.markdown("#### 📥 Download for Review")
    csv = review_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📄 Download '{step_name}' CSV File",
        data=csv,
        file_name=f"manual_review_{step_name.lower().replace(' ', '_')}.csv",
        mime='text/csv',
        help=f"Download the {step_name.lower()} data for manual review and modification"
    )
    
    # Upload section with enhanced styling
    st.markdown("#### 📤 Upload Modified File")
    uploaded_file = st.file_uploader(
        f"Upload the modified '{step_name}' file (optional)", 
        type=['csv'], 
        key=f"uploader_{internal_key}",
        help="Upload your modified CSV file with corrections"
    )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            if 'PARTNO' in uploaded_df.columns and pfep_name in uploaded_df.columns:
                uploaded_df.rename(columns={pfep_name: internal_key, 'PARTNO': 'part_id'}, inplace=True)
                update_map = uploaded_df.set_index('part_id')[internal_key].to_dict()
                df[internal_key] = df['part_id'].map(update_map).fillna(df[internal_key])
                st.success(f"✅ Manual changes for {step_name} applied successfully!")
                st.balloons()
                return df
            else:
                st.error("❌ Upload failed. Ensure the file contains 'PARTNO' and the specific classification column.")
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")
            
    return df

# --- 7. ENHANCED MAIN FUNCTION ---
def main():
    # Enhanced page config
    st.set_page_config(
        layout="wide", 
        page_title="PFEP Analyzer - Smart Manufacturing", 
        page_icon="🏭",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .step-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #6c757d;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 PFEP (Plan For Each Part) Analyzer</h1>
        <p>Smart Manufacturing & Supply Chain Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation and status
    with st.sidebar:
        st.markdown("## 🎯 Process Navigator")
        
        # Initialize session state for step tracking
        if 'steps_completed' not in st.session_state:
            st.session_state.steps_completed = 0
        if 'processor' not in st.session_state:
            st.session_state.processor = None
            
        # Progress indicator in sidebar
        create_progress_indicator(st.session_state.steps_completed, 6)
        
        # Quick stats if processor exists
        if st.session_state.processor:
            st.markdown("### 📊 Quick Stats")
            total_parts = len(st.session_state.processor.data)
            create_metric_card("Total Parts", f"{total_parts:,}", color="#667eea")

    # --- Step 1: Enhanced File Upload Section ---
    display_step_card(
        1, 
        "Data Upload & Configuration", 
        "Upload your BOM files and set production parameters",
        "📁",
        "pending"
    )
    
    with st.expander("🚀 STEP 1: Upload Data Files & Set Parameters", expanded=True):
        # Enhanced upload sections
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # File upload columns with enhanced styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 PBOM Files")
            st.markdown("*Production Bill of Materials*")
            pbom_files = st.file_uploader(
                "Upload PBOM files", 
                accept_multiple_files=True, 
                type=['csv', 'xlsx'], 
                key='pbom',
                help="Upload your Production BOM files in CSV or Excel format"
            )
            if pbom_files:
                st.success(f"✅ {len(pbom_files)} PBOM file(s) uploaded")
                
        with col2:
            st.markdown("### 🔧 MBOM Files")
            st.markdown("*Manufacturing Bill of Materials*")
            mbom_files = st.file_uploader(
                "Upload MBOM files", 
                accept_multiple_files=True, 
                type=['csv', 'xlsx'], 
                key='mbom',
                help="Upload your Manufacturing BOM files in CSV or Excel format"
            )
            if mbom_files:
                st.success(f"✅ {len(mbom_files)} MBOM file(s) uploaded")
                
        with col3:
            st.markdown("### 🏪 Vendor Master")
            st.markdown("*Supplier Information*")
            vendor_file = st.file_uploader(
                "Upload Vendor Master file", 
                type=['csv', 'xlsx'], 
                key='vendor',
                help="Upload your vendor/supplier master data"
            )
            if vendor_file:
                st.success("✅ Vendor file uploaded")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced production parameters section
        st.markdown("### 🏭 Production Parameters")
        st.markdown("*Configure daily production quantities for accurate consumption calculations*")
        
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            st.markdown("#### 🚗 Vehicle Type 1")
            daily_mult_1 = st.number_input(
                "Daily production quantity", 
                min_value=0.0, 
                value=1.0, 
                step=0.1, 
                key="daily_1",
                help="Enter the daily production quantity for Vehicle Type 1"
            )
            
        with param_col2:
            st.markdown("#### 🚙 Vehicle Type 2")
            daily_mult_2 = st.number_input(
                "Daily production quantity", 
                min_value=0.0, 
                value=1.0, 
                step=0.1, 
                key="daily_2",
                help="Enter the daily production quantity for Vehicle Type 2"
            )

    # Enhanced start button
    if st.button("🚀 Start Data Consolidation & Processing", key="start_btn"):
        if not pbom_files and not mbom_files:
            st.error("❌ No BOM data loaded. Please upload at least one PBOM or MBOM file.")
            return

        with st.spinner("🔄 Consolidating data... Please wait."):
            bom_files = {"PBOM": pbom_files, "MBOM": mbom_files}
            master_df, logs = consolidate_data(bom_files, vendor_file, daily_mult_1, daily_mult_2)
            
            # Enhanced log display
            st.markdown("### 📋 Consolidation Report")
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
                st.success("🎉 Data consolidation complete!")
                st.session_state.processor = ComprehensiveInventoryProcessor(master_df.loc[:, ~master_df.columns.duplicated()])
                st.session_state.steps_completed = 1
                
                # Enhanced data preview
                st.markdown("### 📊 Consolidated Data Preview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    create_metric_card("Total Parts", len(master_df), "📦")
                with col2:
                    create_metric_card("Total Columns", len(master_df.columns), "📋")
                with col3:
                    total_qty = master_df['total_qty'].sum() if 'total_qty' in master_df.columns else 0
                    create_metric_card("Total Quantity", f"{total_qty:,.0f}", "📈")
                with col4:
                    net_consumption = master_df['net_daily_consumption'].sum() if 'net_daily_consumption' in master_df.columns else 0
                    create_metric_card("Daily Consumption", f"{net_consumption:,.0f}", "⚡")
                
                with st.expander("View Consolidated Data", expanded=False):
                    st.dataframe(master_df.head(20), use_container_width=True)
            else:
                st.error("❌ Data consolidation failed. Please check the logs.")

    # Processing Steps (only show if processor exists)
    if st.session_state.processor:
        processor = st.session_state.processor
        st.markdown("---")
        st.markdown("## 🔄 Processing Pipeline")

        # Step 2: Family Classification
        display_step_card(
            2, 
            "Family Classification", 
            "Categorize parts into product families using intelligent keyword matching",
            "🏷️",
            "completed" if 'family' in processor.data.columns else "pending"
        )
        
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            if st.button("🔄 Run Family Classification", key="family_btn"):
                with st.spinner("🧠 Analyzing part descriptions and applying family rules..."):
                    processor.run_family_classification()
                    st.success("✅ Automated family classification complete!")
                    st.session_state.steps_completed = max(st.session_state.steps_completed, 2)
                    
                    # Show classification results
                    if 'family' in processor.data.columns:
                        family_counts = processor.data['family'].value_counts()
                        st.markdown("#### 📊 Family Distribution")
                        st.bar_chart(family_counts.head(10))
                        
            if 'family' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'family', 'Family Classification')
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 3: Size Classification
        display_step_card(
            3, 
            "Size Classification", 
            "Classify parts by size (S/M/L/XL) based on dimensions and volume",
            "📏",
            "completed" if 'size_classification' in processor.data.columns else "pending"
        )
        
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            if st.button("📏 Run Size Classification", key="size_btn"):
                with st.spinner("📐 Calculating volumes and classifying by size..."):
                    processor.run_size_classification()
                    st.success("✅ Automated size classification complete!")
                    st.session_state.steps_completed = max(st.session_state.steps_completed, 3)
                    
                    # Show size distribution
                    if 'size_classification' in processor.data.columns:
                        size_counts = processor.data['size_classification'].value_counts()
                        st.markdown("#### 📊 Size Distribution")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(size_counts)
                        with col2:
                            for size, count in size_counts.items():
                                percentage = (count / len(processor.data)) * 100
                                st.metric(f"Size {size}", f"{count} parts", f"{percentage:.1f}%")
                        
            if 'size_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'size_classification', 'Size Classification')
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 4: Part Classification
        display_step_card(
            4, 
            "Part Classification (ABC Analysis)", 
            "Percentage-based classification using advanced pricing analysis",
            "💰",
            "completed" if 'part_classification' in processor.data.columns else "pending"
        )
        
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            if st.button("💰 Run Part Classification", key="part_btn"):
                with st.spinner("📊 Performing percentage-based ABC analysis..."):
                    processor.run_part_classification()
                    st.success("✅ Percentage-based part classification complete!")
                    st.session_state.steps_completed = max(st.session_state.steps_completed, 4)
                    
                    # Enhanced classification results display
                    st.markdown("#### 📈 Classification Ranges & Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 💎 Price Ranges by Class")
                        for class_name, info in processor.classifier.calculated_ranges.items():
                            if info['min'] is not None:
                                create_metric_card(
                                    f"Class {class_name} ({info['count']} parts)", 
                                    f"₹{info['min']:,.0f} - ₹{info['max']:,.0f}",
                                    f"{(info['count']/len(processor.data)*100):.1f}% of total parts"
                                )
                    
                    with col2:
                        if 'part_classification' in processor.data.columns:
                            class_counts = processor.data['part_classification'].value_counts()
                            st.markdown("##### 📊 Distribution Chart")
                            st.bar_chart(class_counts)
                            
            if 'part_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'part_classification', 'Part Classification')
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 5: Distance & Inventory Norms
        display_step_card(
            5, 
            "Distance & Inventory Norms", 
            "Calculate vendor distances and determine optimal inventory levels",
            "🗺️",
            "completed" if 'inventory_classification' in processor.data.columns else "pending"
        )
        
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.markdown("#### 📍 Location Configuration")
            current_pincode = st.text_input(
                "Enter your facility pincode", 
                value="411001", 
                help="This will be used to calculate distances to all vendors"
            )
            
            if st.button("🗺️ Calculate Distance & Inventory Norms", key="distance_btn"):
                with st.spinner("🌍 Calculating distances and inventory norms... This may take several minutes."):
                    processor.run_location_based_norms("Current Location", current_pincode)
                    st.success("✅ Distance calculation and inventory norms complete!")
                    st.session_state.steps_completed = max(st.session_state.steps_completed, 5)
                    
                    # Show distance distribution
                    if 'DISTANCE CODE' in processor.data.columns:
                        st.markdown("#### 📊 Distance Code Distribution")
                        distance_counts = processor.data['DISTANCE CODE'].value_counts().sort_index()
                        distance_labels = {1: "< 50km", 2: "50-250km", 3: "250-750km", 4: "> 750km"}
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(distance_counts)
                        with col2:
                            for code, count in distance_counts.items():
                                if pd.notna(code):
                                    label = distance_labels.get(int(code), f"Code {int(code)}")
                                    percentage = (count / len(processor.data)) * 100
                                    create_metric_card(label, f"{count} vendors", f"{percentage:.1f}%")
                        
            if 'inventory_classification' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'inventory_classification', 'Inventory Classification')
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 6: Warehouse Location Assignment
        display_step_card(
            6, 
            "Warehouse Location Assignment", 
            "Assign optimal warehouse locations based on part characteristics",
            "🏭",
            "completed" if 'wh_loc' in processor.data.columns else "pending"
        )
        
        with st.container():
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            if st.button("🏭 Assign Warehouse Locations", key="warehouse_btn"):
                with st.spinner("🏗️ Analyzing part characteristics and assigning warehouse locations..."):
                    processor.run_warehouse_location_assignment()
                    st.success("✅ Warehouse location assignment complete!")
                    st.session_state.steps_completed = 6
                    
                    # Show warehouse distribution
                    if 'wh_loc' in processor.data.columns:
                        wh_counts = processor.data['wh_loc'].value_counts()
                        st.markdown("#### 🏢 Warehouse Location Distribution")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.bar_chart(wh_counts)
                        with col2:
                            for loc, count in wh_counts.head(6).items():
                                full_name = WAREHOUSE_LOCATION_FULL_FORMS.get(loc, loc)
                                percentage = (count / len(processor.data)) * 100
                                create_metric_card(f"{loc}", f"{count} parts", f"{percentage:.1f}% • {full_name}")
                        
            if 'wh_loc' in processor.data.columns:
                st.session_state.processor.data = manual_review_step(processor.data, 'wh_loc', 'Warehouse Location')
            st.markdown('</div>', unsafe_allow_html=True)

        # Final Report Generation
        st.markdown("---")
        st.markdown("## 🎯 Final Report Generation")
        
        display_step_card(
            "Final", 
            "Generate Comprehensive Report", 
            "Create your formatted Excel report with all analysis results",
            "📊",
            "completed" if st.session_state.steps_completed >= 6 else "pending"
        )
        
        if st.button("📋 Generate Complete Excel Report", key="report_btn"):
            with st.spinner("📊 Generating your comprehensive Excel report..."):
                excel_data = create_formatted_excel_output(processor.data)
                st.success("🎉 Report generated successfully!")
                
                # Final summary metrics
                st.markdown("### 📈 Final Report Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_parts = len(processor.data)
                    create_metric_card("Total Parts Processed", f"{total_parts:,}", "📦")
                    
                with col2:
                    if 'family' in processor.data.columns:
                        families = processor.data['family'].nunique()
                        create_metric_card("Product Families", families, "🏷️")
                    
                with col3:
                    if 'part_classification' in processor.data.columns:
                        classified = processor.data['part_classification'].notna().sum()
                        create_metric_card("Parts Classified", f"{classified:,}", "💎")
                        
                with col4:
                    if 'wh_loc' in processor.data.columns:
                        locations = processor.data['wh_loc'].nunique()
                        create_metric_card("Warehouse Locations", locations, "🏭")
                
                # Enhanced download button
                st.markdown("### 📥 Download Your Report")
                st.download_button(
                    label="📊 Download Complete Excel Report",
                    data=excel_data,
                    file_name=f"PFEP_Analysis_Report_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download the complete PFEP analysis with all classifications and recommendations"
                )
                
                st.balloons()
                st.success("🎊 End-to-end PFEP analysis complete! Your supply chain intelligence report is ready.")

        # Additional insights and recommendations
        if st.session_state.steps_completed >= 3:
            st.markdown("---")
            st.markdown("## 🔍 Insights & Recommendations")
            
            with st.expander("📊 Key Insights from Your Analysis", expanded=False):
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("#### 🏷️ Family Analysis")
                    if 'family' in processor.data.columns:
                        top_families = processor.data['family'].value_counts().head(5)
                        st.markdown("**Top 5 Product Families:**")
                        for family, count in top_families.items():
                            percentage = (count / len(processor.data)) * 100
                            st.markdown(f"• **{family}**: {count} parts ({percentage:.1f}%)")
                    
                    st.markdown("#### 💰 Value Analysis")
                    if 'part_classification' in processor.data.columns:
                        class_dist = processor.data['part_classification'].value_counts()
                        st.markdown("**ABC Classification:**")
                        for class_name in ['AA', 'A', 'B', 'C']:
                            if class_name in class_dist:
                                count = class_dist[class_name]
                                percentage = (count / len(processor.data)) * 100
                                st.markdown(f"• **Class {class_name}**: {count} parts ({percentage:.1f}%)")
                
                with insights_col2:
                    st.markdown("#### 📏 Size Distribution")
                    if 'size_classification' in processor.data.columns:
                        size_dist = processor.data['size_classification'].value_counts()
                        st.markdown("**Size Categories:**")
                        for size, count in size_dist.items():
                            percentage = (count / len(processor.data)) * 100
                            st.markdown(f"• **Size {size}**: {count} parts ({percentage:.1f}%)")
                    
                    st.markdown("#### 🗺️ Distance Analysis")
                    if 'DISTANCE CODE' in processor.data.columns:
                        dist_codes = processor.data['DISTANCE CODE'].value_counts().sort_index()
                        distance_labels = {1: "Local (< 50km)", 2: "Regional (50-250km)", 3: "National (250-750km)", 4: "Far (> 750km)"}
                        st.markdown("**Vendor Distance Distribution:**")
                        for code, count in dist_codes.items():
                            if pd.notna(code):
                                label = distance_labels.get(int(code), f"Code {int(code)}")
                                percentage = (count / len(processor.data)) * 100
                                st.markdown(f"• **{label}**: {count} vendors ({percentage:.1f}%)")

            with st.expander("💡 Strategic Recommendations", expanded=False):
                st.markdown("""
                #### 🎯 Supply Chain Optimization Recommendations
                
                **Inventory Management:**
                - Focus on Class AA and A parts for tighter inventory control
                - Implement vendor-managed inventory for Class C parts
                - Consider consignment inventory for high-volume, low-value items
                
                **Warehouse Organization:**
                - Place fast-moving parts in easily accessible locations
                - Group similar families together for efficient picking
                - Consider automation for high-volume families
                
                **Supplier Strategy:**
                - Develop local suppliers for critical components (Distance Code 1)
                - Negotiate better terms with distant suppliers for non-critical items
                - Consider supplier consolidation opportunities
                
                **Continuous Improvement:**
                - Review classifications quarterly
                - Monitor consumption patterns for trending changes
                - Implement real-time inventory tracking for Class AA parts
                """)

    # Footer with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 30px;">
        <h4 style="color: #2c3e50;">🎉 PFEP Analysis Complete!</h4>
        <p style="color: #6c757d;">Your smart manufacturing intelligence platform has successfully analyzed your supply chain data.</p>
        <p style="color: #6c757d; font-size: 12px;">Powered by Advanced Analytics & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
