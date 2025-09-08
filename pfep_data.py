import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="PFEP Automation Tool",
    page_icon="🏭",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    defaults = {
        'current_step': 1,
        'pfep_df': pd.DataFrame(),
        'pbom_data': pd.DataFrame(),
        'mbom_data': pd.DataFrame(),
        'vendor_data': pd.DataFrame(),
        'pbom_drag_box_qty': 5,
        'mbom_drag_box_qty': 5,
        'station_required': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- HELPER FUNCTIONS ---
def generate_structured_excel(df):
    """Generates a structured Excel file with merged headers and specific formatting."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        # Define formats
        header_format_gray = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1})
        subheader_format_orange = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1})
        subheader_format_blue = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1})

        # Define column structure
        all_columns = [
            'SR.NO', 'PARTNO', 'PART DESCRIPTION', 'Qty/Veh 1', 'Qty/Veh 2', 'TOTAL', 'UOM', 'ST.NO',
            'FAMILY', 'Qty/Veh 1_Daily', 'Qty/Veh 2_Daily', 'NET', 'UNIT PRICE', 'PART CLASSIFICATION',
            'VENDOR CODE', 'VENDOR NAME', 'SUPPLY TYPE', 'PBOM_DRAG_BOX_QTY', 'MBOM_DRAG_BOX_QTY'
        ]
        output_df = pd.DataFrame(columns=all_columns)

        # Map data from the app's DataFrame to the Excel columns
        mapping = { 'PARTNO': 'Part_Number', 'PART DESCRIPTION': 'Description', 'UNIT PRICE': 'Unit_Price', 'VENDOR CODE': 'Vendor' }
        for target, source in mapping.items():
            if source in df.columns:
                output_df[target] = df[source]
        output_df['SR.NO'] = range(1, len(df) + 1)
        output_df['PBOM_DRAG_BOX_QTY'] = st.session_state.pbom_drag_box_qty
        output_df['MBOM_DRAG_BOX_QTY'] = st.session_state.mbom_drag_box_qty if st.session_state.station_required else "N/A"
        
        # Write data and headers to Excel
        output_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=2, header=False, index=False)
        worksheet = writer.sheets['Master Data Sheet']
        worksheet.merge_range('A1:H1', 'PART DETAILS', header_format_gray)
        worksheet.merge_range('I1:N1', 'PRICE & CLASSIFICATION', subheader_format_orange)
        worksheet.merge_range('O1:S1', 'SUPPLY SYSTEM', subheader_format_blue)
        for col_num, value in enumerate(output_df.columns.values):
            worksheet.write(1, col_num, value, header_format_gray)
        worksheet.set_column('A:S', 20)

    return output.getvalue()

# --- STREAMLIT UI STEPS ---
def step1_initial_setup():
    st.header("Step 1: Initial Setup")
    st.info("Configure the number of BOM files you intend to upload in the next step.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("PBOM Configuration")
        st.number_input("How many PBOM files will you upload?", min_value=1, max_value=20, key="pbom_drag_box_qty")
    with col2:
        st.subheader("MBOM Configuration")
        st.checkbox("Do you have MBOM files to upload?", key='station_required')
        if st.session_state.station_required:
            st.number_input("How many MBOM files will you upload?", min_value=1, max_value=20, key="mbom_drag_box_qty")

def step2_upload_data():
    st.header("Step 2: Upload and Merge Data")
    st.subheader("Sub-step 2.1: Upload and Combine BOM Files")
    
    # --- CORRECTED LAYOUT LOGIC ---
    if st.session_state.station_required:
        # If MBOM is needed, create two columns
        col1, col2 = st.columns(2)
        
        # PBOM Upload Section in the first column
        with col1:
            st.info(f"Please upload your {st.session_state.pbom_drag_box_qty} PBOM file(s).")
            with st.container(height=300):
                for i in range(1, st.session_state.pbom_drag_box_qty + 1):
                    st.file_uploader(f"PBOM File {i}", type=['csv', 'xlsx'], key=f"pbom_file_{i}")
            if st.button("Combine All PBOM Files"):
                process_files('pbom', st.session_state.pbom_drag_box_qty)

        # MBOM Upload Section in the second column
        with col2:
            st.info(f"Please upload your {st.session_state.mbom_drag_box_qty} MBOM file(s).")
            with st.container(height=300):
                for i in range(1, st.session_state.mbom_drag_box_qty + 1):
                    st.file_uploader(f"MBOM File {i}", type=['csv', 'xlsx'], key=f"mbom_file_{i}")
            if st.button("Combine All MBOM Files"):
                process_files('mbom', st.session_state.mbom_drag_box_qty)
    else:
        # If MBOM is NOT needed, use a single-column layout for PBOM only
        st.info(f"Please upload your {st.session_state.pbom_drag_box_qty} PBOM file(s).")
        with st.container(height=300):
            for i in range(1, st.session_state.pbom_drag_box_qty + 1):
                st.file_uploader(f"PBOM File {i}", type=['csv', 'xlsx'], key=f"pbom_file_{i}")
        if st.button("Combine All PBOM Files"):
            process_files('pbom', st.session_state.pbom_drag_box_qty)

    st.markdown("---")

    # The rest of the step remains the same...
    # --- SUB-STEP 2.2: CREATE UNIQUE PART MASTER ---
    if not st.session_state.pbom_data.empty:
        st.subheader("Sub-step 2.2: Create Unique Part Master List")
        # Check if MBOM is required and if its data is ready
        mbom_ready = (st.session_state.station_required and not st.session_state.mbom_data.empty) or not st.session_state.station_required
        
        if not mbom_ready:
            st.warning("Please combine your MBOM files before creating the master list.")
        else:
            key_col = st.selectbox("Select the common Part Number column for merging BOMs", options=st.session_state.pbom_data.columns)
            if st.button("Merge BOMs and Find Unique Parts", type="primary"):
                create_unique_part_master(key_col)
        st.markdown("---")

    # --- SUB-STEP 2.3: MERGE VENDOR DATA ---
    if not st.session_state.pfep_df.empty:
        st.subheader("Sub-step 2.3: Upload and Merge Vendor Data")
        merge_vendor_data()

def process_files(bom_type, num_files):
    """Helper function to read and combine uploaded files for either PBOM or MBOM."""
    df_list = []
    for i in range(1, num_files + 1):
        uploaded_file = st.session_state.get(f"{bom_type}_file_{i}")
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
                df_list.append(df)
            except Exception as e:
                st.error(f"Error reading {bom_type.upper()} file {i}: {e}")
                return
    
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        st.session_state[f"{bom_type}_data"] = combined_df
        st.success(f"Combined {len(df_list)} {bom_type.upper()} files into a table with {len(combined_df)} rows.")
        st.dataframe(combined_df.head())
    else:
        st.warning(f"No {bom_type.upper()} files were uploaded to combine.")

def create_unique_part_master(key_col):
    """Merges PBOM and MBOM, then finds unique parts."""
    combined_df = st.session_state.pbom_data
    if not st.session_state.mbom_data.empty:
        combined_df = pd.concat([st.session_state.pbom_data, st.session_state.mbom_data], ignore_index=True)
    
    initial_rows = len(combined_df)
    unique_df = combined_df.drop_duplicates(subset=[key_col], keep='first').copy()
    final_rows = len(unique_df)
    
    st.session_state.pfep_df = unique_df
    st.success(f"Merged BOMs. Found {final_rows} unique parts from an initial {initial_rows} total rows.")
    st.dataframe(st.session_state.pfep_df.head())

def merge_vendor_data():
    """Handles the vendor file upload and merge process."""
    vendor_file = st.file_uploader("Upload Vendor Master File", type=['csv', 'xlsx'], key="vendor_file")
    if vendor_file:
        try:
            st.session_state.vendor_data = pd.read_excel(vendor_file) if vendor_file.name.endswith('.xlsx') else pd.read_csv(vendor_file)
        except Exception as e:
            st.error(f"Error reading vendor file: {e}")
            return
        
        st.write("Vendor Data Preview:")
        st.dataframe(st.session_state.vendor_data.head())
        
        master_key_col = st.selectbox("Select Part Number column in your Master List", options=st.session_state.pfep_df.columns, key="master_key")
        vendor_key_col = st.selectbox("Select Part Number column in your Vendor File", options=st.session_state.vendor_data.columns, key="vendor_key")
        
        if st.button("Merge Vendor Data", type="primary"):
            st.session_state.pfep_df = pd.merge(st.session_state.pfep_df, st.session_state.vendor_data,
                                                left_on=master_key_col, right_on=vendor_key_col, how='left', suffixes=('', '_vendor'))
            st.success("Successfully merged vendor data.")
            st.dataframe(st.session_state.pfep_df.head())

# --- OTHER STEPS ---
def step8_final_output():
    st.header("Step 8: Final PFEP Output")
    if st.session_state.pfep_df.empty: 
        st.warning("No PFEP data to download.")
        return
    st.dataframe(st.session_state.pfep_df)
    excel_data = generate_structured_excel(st.session_state.pfep_df)
    st.download_button(label="📊 Download PFEP Excel (Structured)", data=excel_data,
                        file_name="PFEP_Structured_Output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if st.button("🔄 Reset Application"):
        st.session_state.clear(); st.rerun()

# --- Main Application Logic ---
def main():
    st.title("🏭 PFEP Automation Tool")
    st.markdown("---")
    
    st.progress(st.session_state.current_step / 8)
    st.write(f"Step {st.session_state.current_step} of 8")
    
    col1, _, col3 = st.columns([1, 8, 1])
    if st.session_state.current_step > 1:
        if col1.button("← Previous"):
            st.session_state.current_step -= 1; st.rerun()
    if st.session_state.current_step < 8:
        if col3.button("Next →"):
            st.session_state.current_step += 1; st.rerun()
    
    # Simplified step routing
    steps = {1: step1_initial_setup, 2: step2_upload_data, 8: step8_final_output}
    
    # Use a lambda to show a placeholder for steps not fully implemented
    current_step_func = steps.get(st.session_state.current_step, 
                                  lambda: st.header(f"Step {st.session_state.current_step}: Placeholder"))
    current_step_func()

if __name__ == "__main__":
    main()
