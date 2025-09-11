import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io

# --- SIMPLE STYLING ---
def load_simple_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    
    .big-title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .step-box {
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .big-button {
        font-size: 1.2rem !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        margin: 1rem 0 !important;
    }
    
    .step-number {
        background-color: #1f77b4;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .help-text {
        color: #6c757d;
        font-style: italic;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIMPLIFIED DATA PROCESSING ---
class SimpleProcessor:
    def __init__(self, data):
        self.data = data
        self.geolocator = Nominatim(user_agent=f"simple_pfep_{time.time()}")
    
    def classify_parts_simple(self):
        """Simple ABC classification based on price"""
        if 'unit_price' not in self.data.columns:
            self.data['part_class'] = 'Unknown'
            return
        
        # Convert prices to numeric
        prices = pd.to_numeric(self.data['unit_price'], errors='coerce')
        valid_prices = prices.dropna()
        
        if len(valid_prices) == 0:
            self.data['part_class'] = 'Unknown'
            return
        
        # Simple percentile-based classification
        p80 = valid_prices.quantile(0.8)  # Top 20% are A class
        p50 = valid_prices.quantile(0.5)  # Next 30% are B class
        
        def classify_price(price):
            if pd.isna(price):
                return 'Unknown'
            if price >= p80:
                return 'A (High Value)'
            elif price >= p50:
                return 'B (Medium Value)'
            else:
                return 'C (Low Value)'
        
        self.data['part_class'] = prices.apply(classify_price)
    
    def assign_storage_simple(self):
        """Simple storage assignment based on part description"""
        if 'description' not in self.data.columns:
            self.data['storage_location'] = 'General Storage'
            return
        
        def get_storage(description):
            desc = str(description).upper()
            
            if any(word in desc for word in ['ELECTRICAL', 'BATTERY', 'SENSOR']):
                return 'Electronics Area'
            elif any(word in desc for word in ['WHEEL', 'TIRE', 'TYRE']):
                return 'Heavy Parts Area'
            elif any(word in desc for word in ['BOLT', 'NUT', 'SCREW', 'WASHER']):
                return 'Small Parts Storage'
            elif any(word in desc for word in ['GLASS', 'MIRROR']):
                return 'Fragile Items Area'
            else:
                return 'General Storage'
        
        self.data['storage_location'] = self.data['description'].apply(get_storage)
    
    def calculate_inventory_simple(self, daily_production):
        """Simple inventory calculation"""
        if 'qty_per_vehicle' not in self.data.columns:
            return
        
        # Convert to numeric
        qty = pd.to_numeric(self.data['qty_per_vehicle'], errors='coerce').fillna(0)
        
        # Simple calculation: daily need = qty per vehicle * daily production
        self.data['daily_requirement'] = qty * daily_production
        
        # Simple safety stock: 7 days for A class, 5 for B, 3 for C
        def get_safety_days(part_class):
            if 'A' in str(part_class):
                return 7
            elif 'B' in str(part_class):
                return 5
            else:
                return 3
        
        safety_days = self.data.get('part_class', 'C').apply(get_safety_days)
        self.data['safety_stock'] = self.data['daily_requirement'] * safety_days

# --- SIMPLIFIED COLUMN MAPPING ---
SIMPLE_COLUMN_MAP = {
    'part_id': ['PARTNO', 'Part No', 'Part Number', 'part_number', 'partno'],
    'description': ['PART DESCRIPTION', 'Part Description', 'Description', 'part_desc'],
    'qty_per_vehicle': ['Qty/Veh', 'QTY/VEH', 'Quantity', 'qty', 'quantity_per_vehicle'],
    'unit_price': ['UNIT PRICE', 'Unit Price', 'Price', 'Cost', 'unit_cost'],
    'vendor_name': ['VENDOR NAME', 'Vendor', 'Supplier', 'vendor'],
    'city': ['CITY', 'City', 'Location'],
    'pincode': ['PINCODE', 'Pincode', 'PIN', 'Postal Code']
}

def find_column_match(df_columns, target_variations):
    """Find matching column from variations"""
    for col in df_columns:
        if col in target_variations:
            return col
        # Case-insensitive match
        for variation in target_variations:
            if col.lower().strip() == variation.lower().strip():
                return col
    return None

def standardize_columns(df):
    """Standardize column names"""
    rename_map = {}
    found_columns = []
    
    for standard_name, variations in SIMPLE_COLUMN_MAP.items():
        matched_col = find_column_match(df.columns, variations)
        if matched_col:
            rename_map[matched_col] = standard_name
            found_columns.append(standard_name)
    
    df.rename(columns=rename_map, inplace=True)
    return df, found_columns

def read_file_simple(uploaded_file):
    """Simple file reader"""
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            return None
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

def create_simple_excel(df):
    """Create simple Excel output"""
    output = io.BytesIO()
    
    # Rename columns to business-friendly names
    business_names = {
        'part_id': 'Part Number',
        'description': 'Part Description', 
        'qty_per_vehicle': 'Quantity per Vehicle',
        'daily_requirement': 'Daily Requirement',
        'unit_price': 'Unit Price',
        'part_class': 'ABC Classification',
        'storage_location': 'Recommended Storage',
        'safety_stock': 'Safety Stock Quantity',
        'vendor_name': 'Supplier Name',
        'city': 'Supplier City'
    }
    
    # Select and rename columns that exist
    output_df = df.copy()
    existing_cols = [col for col in business_names.keys() if col in output_df.columns]
    output_df = output_df[existing_cols]
    output_df.rename(columns=business_names, inplace=True)
    
    # Add row numbers
    output_df.insert(0, 'S.No.', range(1, len(output_df) + 1))
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        output_df.to_excel(writer, sheet_name='PFEP Analysis', index=False)
        
        # Simple formatting
        workbook = writer.book
        worksheet = writer.sheets['PFEP Analysis']
        
        # Header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D9E2F3',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(output_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(output_df.columns):
            max_length = max(
                output_df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, min(max_length, 30))
    
    return output.getvalue()

def main():
    # Load simple styling
    load_simple_css()
    
    # Page setup
    st.set_page_config(
        layout="wide", 
        page_title="Simple PFEP Analyzer",
        page_icon="📊"
    )
    
    # Title
    st.markdown('<h1 class="big-title">📊 Simple PFEP Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">Easy inventory planning for your parts - no technical knowledge required!</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Step 1: File Upload
    st.markdown("""
    <div class="step-box">
        <h3><span class="step-number">1</span>Upload Your Files</h3>
        <p>Upload your Excel or CSV files containing part information. The system will automatically find the right columns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 Parts List File")
        parts_file = st.file_uploader(
            "Choose your parts/BOM file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload file with part numbers, descriptions, and quantities"
        )
        
    with col2:
        st.subheader("🏪 Supplier Information (Optional)")
        vendor_file = st.file_uploader(
            "Choose supplier/vendor file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload file with supplier details and locations"
        )
    
    # Step 2: Production Settings
    st.markdown("""
    <div class="step-box">
        <h3><span class="step-number">2</span>Production Settings</h3>
        <p>Tell us how many vehicles/products you make per day</p>
    </div>
    """, unsafe_allow_html=True)
    
    daily_production = st.number_input(
        "How many vehicles/products do you make per day?", 
        min_value=1, 
        value=10, 
        step=1,
        help="This helps calculate daily part requirements"
    )
    
    # Step 3: Process Data
    st.markdown("""
    <div class="step-box">
        <h3><span class="step-number">3</span>Process Your Data</h3>
        <p>Click the button below to analyze your parts and create recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Analyze My Parts", key="process_btn"):
        if not parts_file:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Missing File!</strong><br>
                Please upload at least a parts list file to continue.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Process the data
        with st.spinner("📊 Analyzing your data... This may take a moment."):
            
            # Read main parts file
            df = read_file_simple(parts_file)
            if df is None:
                st.error("Could not read the parts file. Please check the format.")
                return
            
            # Standardize columns
            df, found_cols = standardize_columns(df)
            
            # Check if essential columns exist
            if 'part_id' not in found_cols:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Missing Part Numbers!</strong><br>
                    Could not find part numbers in your file. Make sure your file has a column with part numbers/IDs.
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Merge with vendor file if provided
            if vendor_file:
                vendor_df = read_file_simple(vendor_file)
                if vendor_df is not None:
                    vendor_df, _ = standardize_columns(vendor_df)
                    if 'part_id' in vendor_df.columns:
                        df = pd.merge(df, vendor_df, on='part_id', how='left', suffixes=('', '_vendor'))
                        # Clean up duplicate columns
                        for col in df.columns:
                            if col.endswith('_vendor'):
                                original_col = col.replace('_vendor', '')
                                if original_col in df.columns:
                                    df[original_col] = df[original_col].fillna(df[col])
                                    df.drop(columns=[col], inplace=True)
            
            # Process the data
            processor = SimpleProcessor(df)
            processor.classify_parts_simple()
            processor.assign_storage_simple()
            processor.calculate_inventory_simple(daily_production)
            
            st.session_state.processed_data = processor.data
            
            st.markdown("""
            <div class="success-box">
                <strong>✅ Analysis Complete!</strong><br>
                Successfully analyzed your parts data. Results are shown below.
            </div>
            """, unsafe_allow_html=True)
    
    # Step 4: Show Results
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        st.markdown("""
        <div class="step-box">
            <h3><span class="step-number">4</span>Your Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Parts", len(data))
        
        with col2:
            if 'daily_requirement' in data.columns:
                total_daily = data['daily_requirement'].sum()
                st.metric("Daily Parts Needed", f"{total_daily:,.0f}")
        
        with col3:
            if 'part_class' in data.columns:
                high_value = len(data[data['part_class'].str.contains('A', na=False)])
                st.metric("High-Value Parts", high_value)
        
        with col4:
            if 'storage_location' in data.columns:
                storage_areas = data['storage_location'].nunique()
                st.metric("Storage Areas", storage_areas)
        
        # Show data tables
        tab1, tab2, tab3 = st.tabs(["📋 All Parts", "📊 Summary by Class", "🏪 Storage Areas"])
        
        with tab1:
            st.subheader("Complete Parts List")
            
            # Select columns to show
            display_cols = []
            col_mapping = {
                'part_id': 'Part Number',
                'description': 'Description',
                'qty_per_vehicle': 'Qty per Vehicle',
                'daily_requirement': 'Daily Need',
                'part_class': 'ABC Class',
                'storage_location': 'Storage Area',
                'unit_price': 'Price',
                'vendor_name': 'Supplier'
            }
            
            for col, display_name in col_mapping.items():
                if col in data.columns:
                    display_cols.append(col)
            
            display_data = data[display_cols].copy()
            display_data.rename(columns=col_mapping, inplace=True)
            
            st.dataframe(display_data, use_container_width=True, height=400)
        
        with tab2:
            if 'part_class' in data.columns:
                st.subheader("Parts by ABC Classification")
                
                class_summary = data['part_class'].value_counts().reset_index()
                class_summary.columns = ['ABC Class', 'Number of Parts']
                
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.dataframe(class_summary, use_container_width=True)
                with col_b:
                    st.bar_chart(class_summary.set_index('ABC Class'))
                
                st.markdown("""
                <div class="help-text">
                💡 <strong>What this means:</strong><br>
                • A (High Value): These parts are expensive - keep close control<br>
                • B (Medium Value): Important parts - regular monitoring needed<br>
                • C (Low Value): Cheaper parts - can keep more stock
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            if 'storage_location' in data.columns:
                st.subheader("Recommended Storage Areas")
                
                storage_summary = data['storage_location'].value_counts().reset_index()
                storage_summary.columns = ['Storage Area', 'Number of Parts']
                
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.dataframe(storage_summary, use_container_width=True)
                with col_b:
                    st.bar_chart(storage_summary.set_index('Storage Area'))
        
        # Step 5: Download Results
        st.markdown("""
        <div class="step-box">
            <h3><span class="step-number">5</span>Download Your Report</h3>
            <p>Get your complete analysis in an Excel file that you can share with your team</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Download Excel Report", key="download_btn"):
            excel_data = create_simple_excel(data)
            
            st.download_button(
                label="📄 Click here to download your report",
                data=excel_data,
                file_name=f"PFEP_Analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.balloons()
            
            st.markdown("""
            <div class="success-box">
                <strong>🎉 Report Ready!</strong><br>
                Your Excel report contains all the analysis results with business-friendly column names.
                You can now share this with your team or use it for inventory planning.
            </div>
            """, unsafe_allow_html=True)
    
    # Help section at the bottom
    with st.expander("❓ Need Help? Click here for guidance"):
        st.markdown("""
        ### How to use this tool:
        
        **Step 1: Prepare your files**
        - Your parts file should have columns like: Part Number, Description, Quantity
        - Supplier file (optional) should have: Part Number, Supplier Name, City, Pincode
        - Files can be Excel (.xlsx) or CSV format
        
        **Step 2: Upload and process**
        - Upload your files using the buttons above
        - Enter your daily production quantity
        - Click "Analyze My Parts"
        
        **Step 3: Review results**
        - Check the summary numbers
        - Look at the ABC classification (A = expensive, C = cheap)
        - Review storage recommendations
        
        **Step 4: Download report**
        - Click download to get an Excel file
        - Share with your team or use for planning
        
        ### Common file formats that work:
        - Part Number, Part Description, Qty per Vehicle, Unit Price
        - PARTNO, DESCRIPTION, QTY/VEH, COST
        - Item Code, Item Name, Quantity, Price
        
        ### If you have problems:
        - Make sure your file has part numbers/IDs
        - Check that quantities and prices are numbers (not text)
        - Try with a smaller file first to test
        """)

if __name__ == "__main__":
    main()
