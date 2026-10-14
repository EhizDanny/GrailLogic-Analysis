import pandas as pd
import streamlit as st
from column_config import DEFAULT_COLUMNS, apply_mapping, load_mapping, reset_mapping, save_mapping

st.set_page_config(layout='wide', page_icon=':gear:', page_title='Column Settings')
st.markdown(
    "<h1 style='color: #5409DA; font-size: 38px; text-align: center;'>COLUMN SETTINGS</h1>", unsafe_allow_html=True)
st.info("Set the header each column must have in your CSV files. Uploaded data is matched against these names "
        "(case-insensitive). For example, if your files use 'pansl', set Net P&L USD to 'pansl'.")

current = load_mapping()
with st.form('column_form'):
    head1, head2 = st.columns([1, 1])
    head1.markdown('**Column used by the app**')
    head2.markdown('**Header in your CSV**')
    entries = {}
    for internal in DEFAULT_COLUMNS:
        c1, c2 = st.columns([1, 1])
        c1.markdown(f'`{internal}`')
        entries[internal] = c2.text_input(internal, value=current[internal], label_visibility='collapsed',
                                          key=f'colmap_{internal}')
    save, reset = st.columns([1, 6])
    saved = save.form_submit_button('Save', type='primary')
    reset_clicked = reset.form_submit_button('Reset to defaults')

if saved:
    cleaned = {k: v.strip() for k, v in entries.items()}
    blanks = [k for k, v in cleaned.items() if not v]
    seen = {}
    for k, v in cleaned.items():
        seen.setdefault(v.lower(), []).append(k)
    dupes = [ks for ks in seen.values() if len(ks) > 1]
    if blanks:
        st.error(f'These names cannot be blank: {blanks}')
    elif dupes:
        st.error(f'Two columns cannot share the same header: {dupes}')
    else:
        save_mapping(cleaned)
        st.success('Saved. New uploads will be matched using these names.')
if reset_clicked:
    reset_mapping()
    for internal in DEFAULT_COLUMNS:
        st.session_state.pop(f'colmap_{internal}', None)
    st.rerun()

st.divider()
st.subheader('Check a CSV against the saved settings')
test = st.file_uploader('Upload a CSV to test (not loaded into the analysis)', type='csv', key='colmap_test')
if test:
    _, missing = apply_mapping(pd.read_csv(test, nrows=5))
    if missing:
        st.error('Headers not found in this file: ' + ', '.join(f"'{exp}' (for {internal})" for internal, exp in missing))
    else:
        st.success('All expected columns were found.')
