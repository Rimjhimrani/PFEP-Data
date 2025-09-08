import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import math

# Page configuration
st.set_page_config(
    page_title="PFEP Automation Tool",
    page_icon="🏭",
    layout="wide"
)

# Initialize session state for multi-step workflow and data persistence
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# This will hold the main DataFrame that is built up through the steps
if 'pfep_df' not in st.session_state:
    st.session_state.pfep_df = pd.DataFrame()

# Store uploaded raw dataframes
if 'pbom_data' not in st.session_state:
    st.session_state.pbom_data = pd.DataFrame()

if 'mbom_data' not in st.session_state:
    st.session_state.mbom_data = pd.DataFrame()

if 'vendor_data' not in st.session_state:
    st.session_state.vendor_data = pd.DataFrame()

# --- Helper Functions ---

def calculate_part_classification(value):
    """Calculate part classification based on ABC analysis"""
    # This is a placeholder logic. Real logic would be based on consumption value percentage.
    if value >= 1000: return 'AA'
    elif value >= 500: return 'A'
    elif value >= 100: return 'B'
    else: return 'C'

def get_inventory_classification(part_class):
    """Get inventory classification based on part classification"""
    classifications = {
        'AA': ['A1', 'A2', 'A3', 'A4'], 'A': ['A1', 'A2', 'A3', 'A4'],
        'B': ['B1', 'B2', 'B3', 'B4'], 'C': ['C1', 'C2']
    }
    return classifications.get(part_class, ['C1', 'C2'])[0] # Default to first option

def calculate_rm_days(inventory_class):
    """Calculate RM in Days based on inventory classification"""
    rm_days = {
        'A1': 7, 'A2': 10, 'A3': 15, 'A4': 20,
        'B1': 15, 'B2': 20, 'B3': 25, 'B4': 30,
        'C1': 30, 'C2': 45
    }
    return rm_days.get(inventory_class, 30)

def get_family_keywords():
    """Return predefined family keywords for location selection"""
    return {
        'Engine': ['engine', 'motor', 'piston'], 'Body': ['body', 'panel', 'door'],
        'Electrical': ['wire', 'cable', 'connector'], 'Chassis': ['frame', 'axle', 'suspension'],
        'Interior': ['seat', 'dashboard', 'trim']
    }

def generate_structured_excel(df):
    """
    Generates a structured Excel file with merged headers and specific formatting
    using the logic from the second script.
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book

    # --- DEFINE CELL FORMATS ---
    header_format_gray = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1})
    subheader_format_orange = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1})
    subheader_format_blue = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1})

    # --- DEFINE FINAL EXCEL COLUMN STRUCTURE ---
    # This structure must be maintained for the headers to work correctly.
    all_columns = [
        'SR.NO', 'PARTNO', 'PART DESCRIPTION', 'Qty/Veh 1', 'Qty/Veh 2', 'TOTAL', 'UOM', 'ST.NO',
        'FAMILY', 'Qty/Veh 1_Daily', 'Qty/Veh 2_Daily', 'NET',
        'UNIT PRICE', 'PART CLASSIFICATION',
        'L-MM_Size', 'W-MM_Size', 'H-MM_Size', 'Volume (m^3)', 'SIZE CLASSIFICATION',
        'VENDOR CODE', 'VENDOR NAME', 'VENDOR TYPE', 'CITY', 'STATE', 'COUNTRY', 'PINCODE',
        'PRIMARY PACK TYPE','L-MM_Prim_Pack', 'W-MM_Prim_Pack', 'H-MM_Prim_Pack', 'QTY/PACK_Prim', 'PRIM. PACK LIFESPAN',
        'PRIMARY PACKAGING FACTOR', 'SECONDARY PACK TYPE', 'L-MM_Sec_Pack', 'W-MM_Sec_Pack',
        'H-MM_Sec_Pack', 'NO OF BOXES', 'QTY/PACK_Sec', 'SEC. PACK LIFESPAN', 'ONE WAY/ RETURNABLE',
        'DISTANCE CODE_Pune', 'INVENTORY CLASSIFICATION_Pune', 'RM IN DAYS_Pune', 'RM IN QTY_Pune',
        'RM IN INR_Pune', 'PACKING FACTOR (PF)_Pune', 'NO OF SEC. PACK REQD._Pune',
        'NO OF SEC REQ. AS PER PF_Pithampur', 'DISTANCE CODE_Pithampur', 'INVENTORY CLASSIFICATION_Pithampur',
        'RM IN DAYS_Pithampur', 'RM IN QTY_Pithampur', 'RM IN INR_Pithampur','PACKING FACTOR (PF)_Pithampur', 'NO OF SEC. PACK REQD._Pithampur', 'NO OF SEC REQ. AS PER PF_Pithampur',
        'WH LOC', 'PRIMARY LOCATION ID', 'SECONDARY LOCATION ID', 'OVER FLOW TO BE ALLOTED', 'DOCK NUMBER', 'STACKING FACTOR',
        'SUPPLY TYPE', 'SUPPLY VEH SET', 'SUPPLY STRATEGY', 'SUPPLY CONDITION', 'CONTAINER LINE SIDE',
        'L-MM_Supply', 'W-MM_Supply', 'H-MM_Supply', 'Volume_Supply',
        'QTY/CONTAINER -LS -9M', 'QTY/CONTAINER -LS-12M', 'STORAGE LINE SIDE', 'L-MM_Line',
        'W-MM_Line', 'H-MM_Line', 'Volume_Line', 'CONTAINER / RACK','NO OF TRIPS/DAY', 'INVENTORY LINE SIDE',
        'DRAG BOX QTY' # Added based on user request
    ]

    # --- DATA MAPPING ---
    # Create a new DataFrame with the target structure and map data from the app's DataFrame.
    output_df = pd.DataFrame(columns=all_columns)
    
    # Simple 1-to-1 mapping (assuming column names in uploaded files)
    # This section will need to be adjusted based on the EXACT column names in your source files.
    mapping = {
        'PARTNO': 'Part_Number', 'PART DESCRIPTION': 'Description', 'TOTAL': 'Daily_Consumption',
        'UNIT PRICE': 'Unit_Price', 'PART CLASSIFICATION': 'Classification', 'VENDOR CODE': 'Vendor',
        'RM IN DAYS_Pune': 'RM_Days', 'RM IN QTY_Pune': 'RM_Qty', 'RM IN INR_Pune': 'RM_Value',
        'WH LOC': 'Storage_Location', 'SUPPLY TYPE': 'Supply_Type', 'CONTAINER LINE SIDE': 'Container_Type',
        'ST.NO': 'Station_Number'
    }

    for target_col, source_col in mapping.items():
        if source_col in df.columns:
            output_df[target_col] = df[source_col]

    # Add other data that might have been collected
    output_df['SR.NO'] = range(1, len(df) + 1)
    if 'drag_box_qty' in st.session_state:
        output_df['DRAG BOX QTY'] = st.session_state.drag_box_qty

    # Write data to the Excel sheet
    output_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
    worksheet = writer.sheets['Master Data Sheet']

    # --- WRITE MERGED HEADERS ---
    # Row 1: Main Headers
    worksheet.merge_range('A1:H1', 'PART DETAILS', header_format_gray)
    worksheet.merge_range('I1:L1', 'Daily consumption', subheader_format_orange)
    worksheet.merge_range('M1:N1', 'PRICE & CLASSIFICATION', subheader_format_orange)
    worksheet.merge_range('O1:S1', 'Size & Classification', subheader_format_orange)
    worksheet.merge_range('T1:Z1', 'VENDOR DETAILS', subheader_format_blue)
    worksheet.merge_range('AA1:AO1', 'PACKAGING DETAILS', subheader_format_orange)
    worksheet.merge_range('AP1:AW1', 'PUNE INVENTORY NORM', subheader_format_blue)
    worksheet.merge_range('AX1:BE1', 'PRITHAMPUR INVENTORY NORM', header_format_gray)
    worksheet.merge_range('BF1:BK1', 'WH STORAGE', subheader_format_orange)
    worksheet.merge_range('BL1:BQ1', 'SUPPLY SYSTEM', subheader_format_blue) # Adjusted range for new column
    worksheet.merge_range('BR1:CE1', 'LINE SIDE STORAGE', header_format_gray) # Adjusted range

    # Row 2: Column Headers
    for col_num, value in enumerate(output_df.columns.values):
        # Apply formatting based on column index
        worksheet.write(1, col_num, value, header_format_gray) # Simplified formatting for example

    # Adjust column widths for better readability
    worksheet.set_column('A:A', 6); worksheet.set_column('B:C', 22); worksheet.set_column('D:CE', 18)

    writer.close()
    return output.getvalue()


# --- Main Application UI ---
def main():
    st.title("🏭 PFEP Automation Tool")
    st.markdown("---")
    
    # Progress bar and navigation
    progress = st.session_state.current_step / 8
    st.progress(progress)
    st.write(f"Step {st.session_state.current_step} of 8")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    if st.session_state.current_step > 1:
        if col1.button("← Previous"):
            st.session_state.current_step -= 1
            st.rerun()
    
    if st.session_state.current_step < 8:
        if col3.button("Next →"):
            st.session_state.current_step += 1
            st.rerun()
    
    # Step routing
    if st.session_state.current_step == 1: step1_initial_setup()
    elif st.session_state.current_step == 2: step2_upload_data()
    elif st.session_state.current_step == 3: step3_product_configuration()
    elif st.session_state.current_step == 4: step4_calculations()
    elif st.session_state.current_step == 5: step5_storage_supply()
    elif st.session_state.current_step == 6: step6_container_analysis()
    elif st.session_state.current_step == 7: step7_visualization()
    elif st.session_state.current_step == 8: step8_final_output()

def step1_initial_setup():
    st.header("Step 1: Initial Setup")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Drag Box Configuration")
        st.session_state.drag_box_qty = st.number_input("Drag Box Quantity", min_value=1, max_value=10, value=5)
    with col2:
        st.subheader("Station Configuration")
        st.session_state.station_required = st.checkbox("Station Number Required? (Requires MBOM upload)")

def step2_upload_data():
    st.header("Step 2: Upload & Merge Data Files")
    st.info("Upload PBOM first. Then, MBOM and Vendor data can be merged.")

    # --- UPLOAD ---
    col1, col2 = st.columns(2)
    with col1:
        pbom_file = st.file_uploader("Upload PBOM (Product BOM)", type=['csv', 'xlsx'])
        if pbom_file:
            try:
                df = pd.read_excel(pbom_file) if pbom_file.name.endswith('.xlsx') else pd.read_csv(pbom_file)
                st.session_state.pbom_data = df
                st.session_state.pfep_df = df # Initialize the main DF
                st.success(f"PBOM loaded with {len(df)} rows.")
            except Exception as e: st.error(f"Error loading PBOM: {e}")

        if st.session_state.get('station_required'):
            mbom_file = st.file_uploader("Upload MBOM (Manufacturing BOM)", type=['csv', 'xlsx'])
            if mbom_file:
                try:
                    df = pd.read_excel(mbom_file) if mbom_file.name.endswith('.xlsx') else pd.read_csv(mbom_file)
                    st.session_state.mbom_data = df
                    st.success(f"MBOM loaded with {len(df)} rows.")
                except Exception as e: st.error(f"Error loading MBOM: {e}")

    with col2:
        vendor_file = st.file_uploader("Upload Vendor Data", type=['csv', 'xlsx'])
        if vendor_file:
            try:
                df = pd.read_excel(vendor_file) if vendor_file.name.endswith('.xlsx') else pd.read_csv(vendor_file)
                st.session_state.vendor_data = df
                st.success(f"Vendor data loaded with {len(df)} rows.")
            except Exception as e: st.error(f"Error loading vendor data: {e}")

    # --- MERGE ---
    st.subheader("Merge Data")
    if not st.session_state.pfep_df.empty:
        if not st.session_state.mbom_data.empty and st.session_state.station_required:
            part_col = st.selectbox("Select Part Number column in PBOM/MBOM", options=st.session_state.pfep_df.columns)
            if st.button("Merge MBOM"):
                st.session_state.pfep_df = pd.merge(st.session_state.pfep_df, st.session_state.mbom_data, on=part_col, how='left')
                st.success("MBOM data merged.")

        if not st.session_state.vendor_data.empty:
            part_col_pfep = st.selectbox("Select Part Number column in PFEP", options=st.session_state.pfep_df.columns, key='pfep_part')
            vendor_col_vend = st.selectbox("Select Vendor Code column in Vendor Data", options=st.session_state.vendor_data.columns, key='vend_code')
            if st.button("Merge Vendor Data"):
                # A common scenario is PBOM having vendor code, not part number, for merging with vendor master
                st.session_state.pfep_df = pd.merge(st.session_state.pfep_df, st.session_state.vendor_data, left_on=part_col_pfep, right_on=vendor_col_vend, how='left')
                st.success("Vendor data merged.")

    st.subheader("Current Merged Data")
    st.dataframe(st.session_state.pfep_df.head())

def step3_product_configuration():
    st.header("Step 3: Product Configuration")
    if st.session_state.pfep_df.empty:
        st.warning("Please upload and merge data in Step 2.")
        return
    
    st.info("This step helps define parameters for calculations in the next step.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Consumption Setup")
        st.session_state.daily_vehicles_1 = st.number_input("Daily Production (Veh 1)", min_value=1, value=50)
        st.session_state.daily_vehicles_2 = st.number_input("Daily Production (Veh 2)", min_value=0, value=50)
        
    with col2:
        st.subheader("Part Classification")
        st.write("Part classification will be calculated based on consumption value.")
        st.write("Example: AA (>1000), A (>500), B (>100), C (<100)")
        
def step4_calculations():
    st.header("Step 4: PFEP Calculations")
    if st.session_state.pfep_df.empty:
        st.warning("No data to calculate. Please complete previous steps.")
        return

    df = st.session_state.pfep_df.copy() # Work on a copy
    
    # Column selection for calculations
    st.subheader("Select Columns for Calculation")
    col1, col2, col3 = st.columns(3)
    qty_per_veh_col = col1.selectbox("Quantity Per Vehicle Column", df.columns)
    unit_price_col = col2.selectbox("Unit Price Column", df.columns)
    
    if st.button("Run Calculations"):
        # 1. Daily Consumption
        df['Daily_Consumption'] = df[qty_per_veh_col] * (st.session_state.daily_vehicles_1 + st.session_state.daily_vehicles_2)
        
        # 2. Consumption Value
        df['Consumption_Value'] = df['Daily_Consumption'] * df[unit_price_col]

        # 3. Part Classification
        df['Classification'] = df['Consumption_Value'].apply(calculate_part_classification)

        # 4. Inventory Classification
        df['Inventory_Class'] = df['Classification'].apply(get_inventory_classification)

        # 5. RM Days
        df['RM_Days'] = df['Inventory_Class'].apply(calculate_rm_days)

        # 6. RM Qty & Value
        df['RM_Qty'] = df['Daily_Consumption'] * df['RM_Days']
        df['RM_Value'] = df[unit_price_col] * df['RM_Qty']
        
        # Save back to session state
        st.session_state.pfep_df = df
        st.success("Calculations completed!")

    st.subheader("Calculation Results Preview")
    st.dataframe(st.session_state.pfep_df.head())

def step5_storage_supply():
    st.header("Step 5: Storage & Supply Configuration")
    if st.session_state.pfep_df.empty:
        st.warning("No data found. Please complete previous steps.")
        return
        
    st.info("Assign storage and supply parameters to all parts in the dataset.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Storage")
        storage_loc = st.selectbox("Primary Storage Location", ['HRR', 'CRL', 'MEZ'])
    with col2:
        st.subheader("Supply")
        supply_type = st.selectbox("Supply Type", ['Direct', 'KIT trolley', 'Repacking'])
        container_type = st.selectbox("Container Type", ['Trolley', 'Bin', 'Rack'])
    
    if st.button("Apply Configuration"):
        df = st.session_state.pfep_df.copy()
        df['Storage_Location'] = storage_loc
        df['Supply_Type'] = supply_type
        df['Container_Type'] = container_type
        st.session_state.pfep_df = df
        st.success("Configuration applied to all parts.")
        
    st.dataframe(st.session_state.pfep_df.head())

def step6_container_analysis():
    st.header("Step 6: Container & Trip Analysis")
    st.info("This is a placeholder for more detailed container and logistics calculations.")
    
    trips_per_day = st.number_input("Default Number of Trips per Day", min_value=1, value=4)
    if st.button("Apply Trip Info"):
        df = st.session_state.pfep_df.copy()
        df['Trips_Per_Day'] = trips_per_day
        st.session_state.pfep_df = df
        st.success("Trip information applied.")
    st.dataframe(st.session_state.pfep_df.head())


def step7_visualization():
    st.header("Step 7: Visualization & Analysis")
    if st.session_state.pfep_df.empty or 'Classification' not in st.session_state.pfep_df.columns:
        st.warning("No data to visualize. Please run calculations in Step 4.")
        return

    df = st.session_state.pfep_df
    
    st.subheader("Part Classification Analysis")
    col1, col2 = st.columns(2)
    with col1:
        class_counts = df['Classification'].value_counts().reset_index()
        class_counts.columns = ['Classification', 'Count']
        fig1 = px.pie(class_counts, values='Count', names='Classification', title='Part Count by Classification')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        class_value = df.groupby('Classification')['RM_Value'].sum().reset_index()
        fig2 = px.bar(class_value, x='Classification', y='RM_Value', title='Total RM Value by Classification')
        st.plotly_chart(fig2, use_container_width=True)

def step8_final_output():
    st.header("Step 8: Final PFEP Output")
    if st.session_state.pfep_df.empty:
        st.warning("No final PFEP data to display or download.")
        return

    st.subheader("Final PFEP Data Preview")
    st.dataframe(st.session_state.pfep_df)

    st.subheader("Download Generated PFEP")
    
    # Use the new function to generate the structured excel file in memory
    excel_data = generate_structured_excel(st.session_state.pfep_df)
    
    st.download_button(
        label="📊 Download PFEP Excel (Structured)",
        data=excel_data,
        file_name="PFEP_Structured_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.info("The downloaded Excel file contains the multi-level headers and structure as specified.")
    
    if st.button("🔄 Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
