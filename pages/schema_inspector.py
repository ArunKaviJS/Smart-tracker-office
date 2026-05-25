"""
ui/pages/schema_inspector.py
Debug utility: shows the REAL columns for every KPI-related table.
Use this any time a [DB ERROR] appears referencing an unknown column.
"""

import streamlit as st
from db.connection import run_query, get_table_columns
from ui.components import page_header, section_header

TABLES = [
    "tbl_battery",
    "tbl_device_data",
    "tbl_device_history",
    "tbl_geozones",
    "tbl_trolley",
    "tbl_trolley_maintenance",
]


def render():
    page_header(
        "Schema Inspector",
        "Live view of actual DB table columns — use when debugging [DB ERROR] messages"
    )

    st.info(
        "This page runs `DESCRIBE <table>` against the live database. "
        "All KPI queries adapt automatically to whatever columns exist here."
    )

    for table in TABLES:
        section_header(table)
        cols = get_table_columns(table)

        if not cols:
            st.error(f"Could not read schema for `{table}` — check DB connection or table name.")
            continue

        # Also fetch full DESCRIBE for data types
        describe_df = run_query(f"DESCRIBE `{table}`")
        if not describe_df.empty:
            st.dataframe(describe_df, use_container_width=True, hide_index=True)
        else:
            st.code(", ".join(cols))

    # ── Raw row count per table ───────────────────────────────────────────────
    section_header("Row Counts")
    count_rows = []
    for table in TABLES:
        result = run_query(f"SELECT COUNT(*) AS cnt FROM `{table}`")
        count = result["cnt"].iloc[0] if not result.empty else "ERROR"
        count_rows.append({"Table": table, "Row Count": count})

    import pandas as pd
    st.dataframe(pd.DataFrame(count_rows), use_container_width=True, hide_index=True)