
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Auto Analytics Dashboard",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS - WHITE AND LIGHT PURPLE THEME
# =====================================================
st.markdown("""
<style>

/* Main app background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #FFFFFF !important;
    color: #29213D !important;
}

/* Remove dark header background */
header[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* Make toolbar background transparent */
[data-testid="stToolbar"] {
    background-color: transparent !important;
}

/* Main content spacing */
[data-testid="stMainBlockContainer"] {
    max-width: 1450px;
    padding: 2rem 2rem 3rem 2rem;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #171521 !important;
    font-weight: 700 !important;
}

/* Main title */
h1 {
    font-size: 2.4rem !important;
}

/* Paragraphs and labels */
p, label, .stMarkdown {
    color: #514866;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #F5F0FF !important;
    border-right: 1px solid #E5D9FF !important;
}

/* Sidebar headings and text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #382D50 !important;
}

/* Dropdown and multiselect fields */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #D8C9F4 !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}

/* Selected filter tags */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #E8DEFF !important;
    border: 1px solid #D4C2FF !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #493477 !important;
}

/* Dropdown arrow */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #7055A5 !important;
    fill: #7055A5 !important;
}

/* Dropdown options */
[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[role="option"] {
    color: #382D50 !important;
    background-color: #FFFFFF !important;
}

[role="option"]:hover {
    background-color: #F0E8FF !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background-color: #E8DEFF !important;
    color: #382D50 !important;
    border: 1px solid #D7C5FF !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #DCCBFF !important;
    color: #29213D !important;
}

/* KPI cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E3D8F7 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 4px 14px rgba(82, 55, 130, 0.07);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #625773 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #29213D !important;
    font-weight: 700 !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E3D8F7 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E3D8F7 !important;
    border-radius: 10px !important;
}

/* Dividers */
hr {
    border-color: #E9E1F5 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #716780;
    padding: 25px 0 10px;
    border-top: 1px solid #E9E1F5;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================
DATA_FILE = Path(__file__).resolve().parent / "autos_dataset.csv"


@st.cache_data
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    # Replace common missing-value symbols
    data = data.replace(
        ["?", "NA", "N/A", "na", "n/a", ""],
        np.nan
    )

    # Convert columns to numeric when possible
    for column in data.columns:
        if data[column].dtype == "object":
            converted = pd.to_numeric(
                data[column], errors="coerce"
            )

            if converted.notna().sum() == data[column].notna().sum():
                data[column] = converted

    return data


if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Keep autos_dataset.csv "
        "in the same folder as app.py."
    )
    st.stop()

try:
    df = load_dataset(str(DATA_FILE))
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()


# =====================================================
# FIND IMPORTANT COLUMNS
# =====================================================
def find_column(names, columns):
    for name in names:
        if name in columns:
            return name
    return None


brand_col = find_column(
    ["make", "brand", "company"], df.columns
)

fuel_col = find_column(
    ["fuel-type", "fuel_type", "fuel"], df.columns
)

price_col = find_column(
    ["price", "selling-price", "selling_price"], df.columns
)

horsepower_col = find_column(
    ["horsepower", "hp"], df.columns
)

engine_col = find_column(
    ["engine-size", "engine_size", "enginesize"], df.columns
)


# =====================================================
# DASHBOARD HEADER
# =====================================================
st.markdown("""
<h1 style="text-align:center; color:#171521;">
    🚘 Auto Analytics Dashboard
</h1>
<p style="text-align:center; font-size:18px; color:#514866;">
    Automobile Data Analysis &amp; Insights
</p>
""", unsafe_allow_html=True)

st.divider()


# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.title("🔎 Dashboard Filters")
st.sidebar.caption("Choose options to explore the dataset.")

all_brands = (
    sorted(df[brand_col].dropna().astype(str).unique().tolist())
    if brand_col else []
)

all_fuels = (
    sorted(df[fuel_col].dropna().astype(str).unique().tolist())
    if fuel_col else []
)

# Reset button: update state before widgets are rendered
if st.sidebar.button("Reset Filters", use_container_width=True):
    st.session_state["reset_auto_filters"] = True
    st.rerun()

if st.session_state.pop("reset_auto_filters", False):
    st.session_state["brand_filter"] = all_brands.copy()
    st.session_state["fuel_filter"] = all_fuels.copy()

# Initialize filter values safely
if "brand_filter" not in st.session_state:
    st.session_state["brand_filter"] = all_brands.copy()

if "fuel_filter" not in st.session_state:
    st.session_state["fuel_filter"] = all_fuels.copy()

# Keep selected values valid
st.session_state["brand_filter"] = [
    item for item in st.session_state["brand_filter"]
    if item in all_brands
]

st.session_state["fuel_filter"] = [
    item for item in st.session_state["fuel_filter"]
    if item in all_fuels
]

filtered_df = df.copy()

if brand_col:
    selected_brands = st.sidebar.multiselect(
        "Select Brand",
        options=all_brands,
        key="brand_filter",
        placeholder="Choose brands..."
    )

    filtered_df = filtered_df[
        filtered_df[brand_col].astype(str).isin(selected_brands)
    ]

if fuel_col:
    selected_fuels = st.sidebar.multiselect(
        "Select Fuel Type",
        options=all_fuels,
        key="fuel_filter",
        placeholder="Choose fuel types..."
    )

    filtered_df = filtered_df[
        filtered_df[fuel_col].astype(str).isin(selected_fuels)
    ]

st.sidebar.divider()
st.sidebar.caption(f"Total records: {len(df):,}")
st.sidebar.caption(f"Filtered records: {len(filtered_df):,}")


# =====================================================
# HANDLE EMPTY FILTER RESULTS
# =====================================================
if filtered_df.empty:
    st.warning(
        "No records match your selected filters. "
        "Please select another brand or fuel type."
    )
    st.stop()


# =====================================================
# KEY PERFORMANCE INDICATORS
# =====================================================
st.subheader("📊 Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)

k1.metric("🚘 Total Cars", f"{len(filtered_df):,}")

k2.metric(
    "🏭 Unique Brands",
    f"{filtered_df[brand_col].nunique():,}" if brand_col else "N/A"
)

if price_col:
    prices = pd.to_numeric(
        filtered_df[price_col], errors="coerce"
    )
    avg_price = prices.mean()

    k3.metric(
        "💰 Average Price",
        f"${avg_price:,.0f}" if pd.notna(avg_price) else "N/A"
    )
else:
    k3.metric("💰 Average Price", "N/A")

if horsepower_col:
    horsepower = pd.to_numeric(
        filtered_df[horsepower_col], errors="coerce"
    )
    avg_hp = horsepower.mean()

    k4.metric(
        "⚙️ Average Horsepower",
        f"{avg_hp:.1f}" if pd.notna(avg_hp) else "N/A"
    )
else:
    k4.metric("⚙️ Average Horsepower", "N/A")


# =====================================================
# DATASET PREVIEW
# =====================================================
st.divider()
st.subheader("📄 Dataset Preview")

st.caption(
    f"Showing up to 20 records from "
    f"{len(filtered_df):,} filtered records."
)

st.dataframe(
    filtered_df.head(20),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="⬇️ Download Filtered Dataset",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_automobile_data.csv",
    mime="text/csv"
)


# =====================================================
# VISUALIZATIONS
# =====================================================
st.divider()
st.subheader("📈 Data Visualizations")

left, right = st.columns(2)


# Price distribution
if price_col:
    price_values = pd.to_numeric(
        filtered_df[price_col], errors="coerce"
    ).dropna()

    with left:
        st.markdown("#### Price Distribution")

        if not price_values.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            ax.hist(
                price_values,
                bins=20,
                color="#CDBCF5",
                edgecolor="#8B7BA8"
            )

            ax.set_xlabel("Price")
            ax.set_ylabel("Number of Cars")
            ax.set_title("Distribution of Automobile Prices")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No valid price data available.")


# Fuel type distribution
if fuel_col:
    fuel_values = (
        filtered_df[fuel_col].dropna().astype(str).value_counts()
    )

    with right:
        st.markdown("#### Fuel Type Distribution")

        if not fuel_values.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("white")

            ax.pie(
                fuel_values.values,
                labels=fuel_values.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[
                    "#CDBCF5", "#E7DDFB",
                    "#B7A4E8", "#A9C5EF"
                ]
            )

            ax.set_title("Cars by Fuel Type")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No fuel type data available.")


# Average price by brand
if brand_col and price_col:
    st.markdown("#### Average Price by Brand")

    brand_data = filtered_df[[brand_col, price_col]].copy()
    brand_data[price_col] = pd.to_numeric(
        brand_data[price_col], errors="coerce"
    )
    brand_data = brand_data.dropna()

    if not brand_data.empty:
        average_by_brand = (
            brand_data.groupby(brand_col)[price_col]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.barh(
            average_by_brand.index.astype(str),
            average_by_brand.values,
            color="#CDBCF5"
        )

        ax.set_xlabel("Average Price")
        ax.set_ylabel("Brand")
        ax.set_title("Average Price by Brand")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough valid data for this chart.")


# Engine size vs price
if engine_col and price_col:
    st.markdown("#### Engine Size vs Price")

    scatter = filtered_df[[engine_col, price_col]].copy()
    scatter[engine_col] = pd.to_numeric(
        scatter[engine_col], errors="coerce"
    )
    scatter[price_col] = pd.to_numeric(
        scatter[price_col], errors="coerce"
    )
    scatter = scatter.dropna()

    if not scatter.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.scatter(
            scatter[engine_col],
            scatter[price_col],
            color="#A68BDD",
            alpha=0.8
        )

        ax.set_xlabel("Engine Size")
        ax.set_ylabel("Price")
        ax.set_title("Engine Size vs Price")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough valid data for this chart.")


# =====================================================
# CORRELATION ANALYSIS
# =====================================================
st.divider()
st.subheader("🔗 Correlation Analysis")

numeric_df = filtered_df.select_dtypes(include=np.number)

if numeric_df.shape[1] >= 2:
    correlation = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    image = ax.imshow(
        correlation,
        cmap="Purples",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(correlation.columns)))
    ax.set_yticklabels(correlation.columns)

    for row in range(len(correlation.columns)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]

            if pd.notna(value):
                ax.text(
                    col, row, f"{value:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="#29213D"
                )

    ax.set_title("Correlation Between Numeric Features")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info(
        "At least two numeric columns are needed "
        "for correlation analysis."
    )


# =====================================================
# DATASET INFORMATION
# =====================================================
st.divider()
st.subheader("📌 Dataset Information")

i1, i2, i3 = st.columns(3)

i1.metric("Total Rows", f"{len(filtered_df):,}")
i2.metric("Total Columns", f"{filtered_df.shape[1]:,}")
i3.metric(
    "Missing Values",
    f"{int(filtered_df.isna().sum().sum()):,}"
)

with st.expander("View Missing Values by Column"):
    missing = filtered_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("No missing values found.")
    else:
        missing_table = missing.rename("Missing Values").reset_index()
        missing_table.columns = ["Column", "Missing Values"]

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True
        )

with st.expander("View Column Details"):
    details = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": [
            str(filtered_df[col].dtype)
            for col in filtered_df.columns
        ],
        "Non-Missing Values": [
            int(filtered_df[col].notna().sum())
            for col in filtered_df.columns
        ],
        "Missing Values": [
            int(filtered_df[col].isna().sum())
            for col in filtered_df.columns
        ]
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <strong>Auto Analytics Dashboard</strong><br>
    Built with Python, Pandas, Matplotlib and Streamlit
</div>
""", unsafe_allow_html=True)
