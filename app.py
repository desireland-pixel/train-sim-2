# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
from pathlib import Path
from modules.assignment import assign_packages


st.set_page_config(layout="wide", page_title="Train-Warehouse Simulation")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------
# Load data (CSVs)
# -------------------------
def load_csv(filename):
    path = DATA_DIR / filename
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty or len(df.columns) == 0:
                return pd.DataFrame()
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

trains = load_csv("trains.csv")
warehouses = load_csv("warehouses.csv")
packages = load_csv("packages.csv")
persons = load_csv("persons.csv")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Simulation Settings")
max_packages_per_person = st.sidebar.number_input("Max packages a person can carry", 1, 10, 5)
num_people = st.sidebar.number_input("If persons.csv missing, auto-create N persons", 1, 50, 10)
current_time = st.sidebar.number_input("Current time (minutes)", 0, 60, 0)

# -------------------------
# Orders per train inputs
# -------------------------
st.sidebar.markdown("### Orders per Train")
order_T1 = st.sidebar.number_input("T1 Orders", 0, 20, 0)
order_T2 = st.sidebar.number_input("T2 Orders", 0, 20, 0)
order_T3 = st.sidebar.number_input("T3 Orders", 0, 20, 0)
order_T4 = st.sidebar.number_input("T4 Orders", 0, 20, 0)
order_T5 = st.sidebar.number_input("T5 Orders", 0, 20, 0)
train_orders = [order_T1, order_T2, order_T3, order_T4, order_T5]

# -------------------------
# Generate Packages Button
# -------------------------
if st.sidebar.button("Generate Packages from Orders"):
    gen_packages = []

    # For each train, use the manual order inputs from sidebar
    for i, train_id in enumerate(trains.train_id, 1):
        n_orders = train_orders[i - 1]  # take order count directly from sidebar
        if n_orders > 0:
            start_time = int(trains.loc[trains.train_id == train_id, "start_time"].values[0])
            for j in range(1, n_orders + 1):
                pkg_id = f"{i:02d}{j:02d}"  # 0101, 0102 ... etc.
                warehouse_id = np.random.choice(warehouses.warehouse_id)  # random W1–W6
                gen_packages.append({
                    "package_id": pkg_id,
                    "warehouse_id": warehouse_id,
                    "generated_time": start_time - 10
                })

    # Convert only if we actually generated packages
    if gen_packages:
        packages = pd.DataFrame(gen_packages)
        st.session_state["packages"] = packages
        packages.to_csv(DATA_DIR / "packages.csv", index=False)
        st.session_state["pkg_text"] = packages[["package_id", "warehouse_id", "generated_time"]]
    else:
        st.warning("No orders entered — no packages generated.")
        st.session_state.pop("packages", None)
        st.session_state.pop("pkg_text", None)

# -------------------------
# Page title
# -------------------------
st.title("🚉 Train–Warehouse Simulation")
st.markdown(f"**Simulation Time: {current_time} min**")

# -------------------------
# Simulation visuals
# -------------------------
fig = go.Figure()

# Warehouses
fig.add_trace(go.Scatter(
    x=warehouses.x, y=warehouses.y,
    mode="markers+text",
    text=warehouses.warehouse_id,
    name="Warehouses",
    marker=dict(size=15, color="green", symbol="square"),
    textposition="top center",
    textfont=dict(color="black")
))

# Platforms (5 fixed)
platforms = pd.DataFrame({
    'platform': [1,2,3,4,5],
    'x': [200,200,200,200,200],
    'y': [150,100,50,0,-50]
})
fig.add_trace(go.Scatter(
    x=platforms.x, y=platforms.y,
    mode="markers+text",
    text=[f"P{i}" for i in platforms.platform],
    name="Platforms",
    marker=dict(size=18, color="blue")
))

# Trains movement
train_positions = []
for _, r in trains.iterrows():
    if current_time < r.start_time:
        x, y = r.x_source, r.y_source
    elif current_time > r.arrive_time:
        x, y = r.x_platform, r.y_platform
    else:
        frac = (current_time - r.start_time) / (r.arrive_time - r.start_time)
        x = r.x_source + frac * (r.x_platform - r.x_source)
        y = r.y_source + frac * (r.y_platform - r.y_source)
    train_positions.append((r.train_id, x, y))

fig.add_trace(go.Scatter(
    x=[x for _,x,_ in train_positions],
    y=[y for _,_,y in train_positions],
    text=[tid for tid,_,_ in train_positions],
    mode="markers+text",
    name="Trains",
    marker=dict(size=20, color="red"),
    textfont=dict(color="white")
))

# -------------------------
# Packages display
# -------------------------
base_hour = 9
base_minute = 0

if "packages" in st.session_state:
    packages = st.session_state["packages"]
    
# Group packages by warehouse to arrange them neatly
offset_x = 12
col_spacing = 12
row_spacing = 25
max_cols = 5

for wh_id, group in packages.groupby("warehouse_id"):
    wh = warehouses[warehouses.warehouse_id == wh_id].iloc[0]

    for idx, (_, pkg) in enumerate(group.iterrows()):
        if current_time >= pkg.generated_time:
            col = idx % max_cols
            row = idx // max_cols

            # position right of warehouse
            x = wh.x + offset_x + col * col_spacing
            y = wh.y + row * row_spacing

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                text=[pkg.package_id],
                textposition="bottom center",
                marker=dict(size=8, color="#D2B48C", symbol="square",
                            line=dict(color="black", width=0.25)),
                name="Packages",
                showlegend=(len(fig.data) == 3)
            ))

# -------------------------
# Clock on top right
# -------------------------
total_minutes = base_minute + current_time
display_hour = base_hour + total_minutes // 60
display_minute = total_minutes % 60
clock_str = f"{display_hour:02d}:{display_minute:02d}"
st.markdown(f"""
<div style='text-align: right; font-size:48px;'>
    ⏰ {clock_str}
</div>
""", unsafe_allow_html=True)

# Show chart
st.plotly_chart(fig, use_container_width=True)
# Fix chart axis ranges
fig.update_xaxes(range=[0, 500])
fig.update_yaxes(range=[-100, 200])
fig.update_layout(autosize=False)

# -------------------------
# Show package text summary
# -------------------------
if "pkg_text" in st.session_state:
    pkg_text = st.session_state["pkg_text"].copy()

    # Format generated_time as HH:MM (base 09:00)
    base_hour = 9
    pkg_text["generated_time"] = pkg_text["generated_time"].apply(
        lambda t: f"{base_hour + t // 60:02d}:{t % 60:02d}"
    )

    st.markdown("**Generated Packages:**")
    st.dataframe(pkg_text)

# -------------------------
# Assign Packages Button & Results
# -------------------------
st.sidebar.markdown("### Assignment")
if st.sidebar.button("Assign Packages"):
    # get packages df (use session_state if newly generated)
    if "packages" not in st.session_state:
        st.warning("No packages available. Generate packages first.")
    else:
        pkgs = st.session_state["packages"].copy()
        # Ensure columns: package_id, warehouse_id
        if 'package_id' not in pkgs.columns or 'warehouse_id' not in pkgs.columns:
            st.error("packages table missing required columns ('package_id', 'warehouse_id').")
        else:
            # call assignment module
            assignments_df, summary_df, per_train_detail, meta = assign_packages(
                pkgs, trains, warehouses, int(max_packages_per_person)
            )
            st.session_state['assignments_df'] = assignments_df
            st.session_state['summary_df'] = summary_df
            st.session_state['per_train_detail'] = per_train_detail
            st.session_state['assignment_meta'] = meta

            st.success(f"Assigned {meta['total_packages']} packages -> {meta['total_persons']} persons")
            st.markdown("**Assignment Summary (train × warehouse):**")
            # show summary as a nicer table with train as first col
            st.dataframe(summary_df.fillna(0).set_index('train_id'))

            # Allow selecting a train to show details (drill-down)
            train_options = list(summary_df['train_id'])
            sel_train = st.selectbox("Select Train to view details", options=train_options)
            if sel_train:
                detail = per_train_detail.get(sel_train, pd.DataFrame())
                if detail.empty:
                    st.info("No assignment details for selected train.")
                else:
                    # show compact detail (warehouse, person, count)
                    st.markdown(f"**Details for {sel_train}:**")
                    # Show packages list if desired
                    # create a copy for display where packages are joined as comma string
                    detail_disp = detail.copy()
                    detail_disp['packages'] = detail_disp['packages'].apply(lambda lst: ",".join(lst))
                    detail_disp = detail_disp[['warehouse', 'person', 'packages', 'count']]
                    detail_disp = detail_disp.rename(columns={'warehouse': 'Warehouse', 'person': 'Person', 'packages': 'Package IDs', 'count': 'Count'})
                    st.dataframe(detail_disp)


# -------------------------
# Info text
# -------------------------
st.info("Use the button in the sidebar to move time forward or backward.")
