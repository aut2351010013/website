import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insight Spark - AI Data Analyst",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 25px;
    }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 135px;
    }

    .kpi-title {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .kpi-value {
        color: #111827;
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-description {
        color: #94a3b8;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    /* Insight box */
    .insight-box {
        background: #eef6ff;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 18px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .insight-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e3a8a;
    }

    .insight-text {
        font-size: 14px;
        color: #334155;
        margin-top: 6px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Hide default menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def money(value):
    """Format numbers as currency."""
    if pd.isna(value):
        return "$0"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"

    return f"${value:,.2f}"


def number_format(value):
    """Format large numbers."""
    if pd.isna(value):
        return "0"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def find_column(df, possible_names):
    """
    Find a dataframe column using multiple possible names.
    """

    normalized = {
        str(col).strip().lower().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    for name in possible_names:

        key = (
            name.lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if key in normalized:
            return normalized[key]

    # Partial matching
    for col in df.columns:

        normalized_col = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        for name in possible_names:

            normalized_name = (
                name.lower()
                .replace(" ", "")
                .replace("_", "")
            )

            if normalized_name in normalized_col:
                return col

    return None


def create_demo_data():
    """
    Create sample sales data when no CSV is uploaded.
    """

    np.random.seed(42)

    categories = [
        "Technology",
        "Furniture",
        "Office Supplies",
        "Accessories"
    ]

    regions = [
        "North",
        "South",
        "East",
        "West"
    ]

    products = [
        "Laptop",
        "Monitor",
        "Keyboard",
        "Mouse",
        "Desk",
        "Chair",
        "Printer",
        "Notebook",
        "Headset",
        "Tablet"
    ]

    dates = pd.date_range(
        start="2025-01-01",
        end="2025-12-31",
        periods=500
    )

    data = []

    for i in range(500):

        sales = np.random.randint(100, 5000)

        cost = sales * np.random.uniform(
            0.45,
            0.85
        )

        profit = sales - cost

        data.append({
            "Order ID": f"ORD-{1000 + i}",
            "Date": dates[i],
            "Product": np.random.choice(products),
            "Category": np.random.choice(categories),
            "Region": np.random.choice(regions),
            "Sales": round(sales, 2),
            "Profit": round(profit, 2),
            "Quantity": np.random.randint(1, 10),
            "Customer": f"Customer {np.random.randint(1, 101)}"
        })

    return pd.DataFrame(data)


def prepare_data(df):

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Strip column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Detect important columns
    date_col = find_column(
        df,
        [
            "Date",
            "Order Date",
            "Transaction Date",
            "Sales Date"
        ]
    )

    sales_col = find_column(
        df,
        [
            "Sales",
            "Revenue",
            "Amount",
            "Total Sales",
            "Price"
        ]
    )

    profit_col = find_column(
        df,
        [
            "Profit",
            "Net Profit",
            "Gross Profit"
        ]
    )

    category_col = find_column(
        df,
        [
            "Category",
            "Product Category",
            "Type"
        ]
    )

    region_col = find_column(
        df,
        [
            "Region",
            "Area",
            "Location",
            "Territory"
        ]
    )

    product_col = find_column(
        df,
        [
            "Product",
            "Product Name",
            "Item"
        ]
    )

    quantity_col = find_column(
        df,
        [
            "Quantity",
            "Qty",
            "Units"
        ]
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if date_col:

        df["Date"] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

    else:

        df["Date"] = pd.date_range(
            start="2025-01-01",
            periods=len(df),
            freq="D"
        )

    # --------------------------------------------------------
    # SALES
    # --------------------------------------------------------

    if sales_col:

        df["Sales"] = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        ).fillna(0)

    else:

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) > 0:

            df["Sales"] = pd.to_numeric(
                df[numeric_columns[0]],
                errors="coerce"
            ).fillna(0)

        else:

            df["Sales"] = 0

    # --------------------------------------------------------
    # PROFIT
    # --------------------------------------------------------

    if profit_col:

        df["Profit"] = pd.to_numeric(
            df[profit_col],
            errors="coerce"
        ).fillna(0)

    else:

        # Estimate profit if profit column does not exist
        df["Profit"] = df["Sales"] * 0.20

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    if quantity_col:

        df["Quantity"] = pd.to_numeric(
            df[quantity_col],
            errors="coerce"
        ).fillna(1)

    else:

        df["Quantity"] = 1

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if category_col:

        df["Category"] = (
            df[category_col]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        df["Category"] = "All Products"

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    if region_col:

        df["Region"] = (
            df[region_col]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        df["Region"] = "All Regions"

    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    if product_col:

        df["Product"] = (
            df[product_col]
            .fillna("Unknown")
            .astype(str)
        )

    else:

        df["Product"] = "Product"

    return df


# ============================================================
# AI STYLE INSIGHTS
# ============================================================

def generate_insights(df):

    insights = []

    if df.empty:
        return insights

    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()

    # --------------------------------------------------------
    # PROFIT MARGIN
    # --------------------------------------------------------

    if total_sales > 0:

        margin = (
            total_profit /
            total_sales
        ) * 100

        if margin >= 25:

            insights.append(
                f"Strong profitability: the overall "
                f"profit margin is approximately {margin:.1f}%."
            )

        elif margin >= 10:

            insights.append(
                f"The business is maintaining a moderate "
                f"profit margin of approximately {margin:.1f}%."
            )

        else:

            insights.append(
                f"Profitability needs attention because "
                f"the overall margin is only {margin:.1f}%."
            )

    # --------------------------------------------------------
    # BEST CATEGORY
    # --------------------------------------------------------

    if "Category" in df.columns:

        category_sales = (
            df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if len(category_sales) > 0:

            best_category = category_sales.index[0]

            best_value = category_sales.iloc[0]

            percentage = (
                best_value /
                total_sales *
                100
                if total_sales > 0
                else 0
            )

            insights.append(
                f"{best_category} is the strongest category, "
                f"generating {money(best_value)} in sales "
                f"({percentage:.1f}% of total sales)."
            )

    # --------------------------------------------------------
    # BEST REGION
    # --------------------------------------------------------

    if "Region" in df.columns:

        region_sales = (
            df.groupby("Region")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if len(region_sales) > 0:

            best_region = region_sales.index[0]

            insights.append(
                f"{best_region} is currently the highest-performing "
                f"region based on sales."
            )

    # --------------------------------------------------------
    # BEST PRODUCT
    # --------------------------------------------------------

    if "Product" in df.columns:

        product_sales = (
            df.groupby("Product")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if len(product_sales) > 0:

            best_product = product_sales.index[0]

            insights.append(
                f"{best_product} is the top-selling product "
                f"in the current dataset."
            )

    return insights


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✨ Insight Spark</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Data Analytics & Business Intelligence Dashboard'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ✨ Insight Spark"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 📂 Data Source"
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a sales or business dataset."
    )

    st.markdown("---")

    st.markdown(
        "### 📊 Dashboard"
    )

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Analytics",
            "Data Explorer",
            "AI Insights"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption(
        "Upload a CSV file to analyze your own data."
    )


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        df = prepare_data(df)

        st.sidebar.success(
            f"Loaded: {uploaded_file.name}"
        )

    except Exception as e:

        st.error(
            f"Unable to read the CSV file: {e}"
        )

        st.stop()

else:

    df = create_demo_data()

    st.sidebar.info(
        "Demo data is being displayed. "
        "Upload your CSV to analyze your own data."
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:

    st.markdown("---")

    st.markdown(
        "### 🔎 Filters"
    )

    # Category filter

    categories = sorted(
        df["Category"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_categories = st.multiselect(
        "Category",
        categories,
        default=categories
    )

    # Region filter

    regions = sorted(
        df["Region"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_regions = st.multiselect(
        "Region",
        regions,
        default=regions
    )

    # Date filter

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    if pd.notna(min_date) and pd.notna(max_date):

        date_range = st.date_input(
            "Date Range",
            value=(
                min_date.date(),
                max_date.date()
            )
        )

    else:

        date_range = None


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_categories:

    filtered_df = filtered_df[
        filtered_df["Category"].isin(
            selected_categories
        )
    ]

if selected_regions:

    filtered_df = filtered_df[
        filtered_df["Region"].isin(
            selected_regions
        )
    ]

if date_range and len(date_range) == 2:

    start_date = pd.to_datetime(
        date_range[0]
    )

    end_date = pd.to_datetime(
        date_range[1]
    )

    filtered_df = filtered_df[
        (filtered_df["Date"] >= start_date)
        &
        (filtered_df["Date"] <= end_date)
    ]


# ============================================================
# CALCULATE KPIs
# ============================================================

total_sales = filtered_df["Sales"].sum()

total_profit = filtered_df["Profit"].sum()

total_orders = len(filtered_df)

total_quantity = filtered_df["Quantity"].sum()

average_order_value = (
    total_sales / total_orders
    if total_orders > 0
    else 0
)

profit_margin = (
    total_profit /
    total_sales *
    100
    if total_sales > 0
    else 0
)


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown("## 📊 Business Overview")

    st.caption(
        f"Showing {len(filtered_df):,} transactions"
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">TOTAL SALES</div>
                <div class="kpi-value">{money(total_sales)}</div>
                <div class="kpi-description">
                    Overall revenue
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">TOTAL PROFIT</div>
                <div class="kpi-value">{money(total_profit)}</div>
                <div class="kpi-description">
                    Net business profit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">TOTAL ORDERS</div>
                <div class="kpi-value">{number_format(total_orders)}</div>
                <div class="kpi-description">
                    Transactions
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">PROFIT MARGIN</div>
                <div class="kpi-value">{profit_margin:.1f}%</div>
                <div class="kpi-description">
                    Overall margin
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # SECOND KPI ROW
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Average Order Value",
            money(average_order_value)
        )

    with col2:

        st.metric(
            "Units Sold",
            number_format(total_quantity)
        )

    with col3:

        st.metric(
            "Products",
            filtered_df["Product"].nunique()
        )

    with col4:

        st.metric(
            "Categories",
            filtered_df["Category"].nunique()
        )

    st.markdown("---")

    # --------------------------------------------------------
    # SALES TREND
    # --------------------------------------------------------

    st.markdown("### 📈 Sales & Profit Trend")

    daily = (
        filtered_df
        .groupby("Date")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum")
        )
        .reset_index()
        .sort_values("Date")
    )

    if not daily.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Sales"],
                mode="lines",
                name="Sales",
                line=dict(width=3)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Profit"],
                mode="lines",
                name="Profit",
                line=dict(width=3)
            )
        )

        fig.update_layout(
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20
            ),
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CATEGORY + REGION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🏷️ Sales by Category")

        category_df = (
            filtered_df
            .groupby("Category")["Sales"]
            .sum()
            .reset_index()
            .sort_values(
                "Sales",
                ascending=False
            )
        )

        fig_category = px.bar(
            category_df,
            x="Category",
            y="Sales",
            text_auto=".2s",
            template="plotly_white"
        )

        fig_category.update_layout(
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    with col2:

        st.markdown("### 🌎 Sales by Region")

        region_df = (
            filtered_df
            .groupby("Region")["Sales"]
            .sum()
            .reset_index()
            .sort_values(
                "Sales",
                ascending=False
            )
        )

        fig_region = px.pie(
            region_df,
            names="Region",
            values="Sales",
            hole=0.55
        )

        fig_region.update_layout(
            height=350
        )

        st.plotly_chart(
            fig_region,
            use_container_width=True
        )


# ============================================================
# ANALYTICS PAGE
# ============================================================

elif page == "Analytics":

    st.markdown("## 📈 Advanced Analytics")

    # --------------------------------------------------------
    # CATEGORY PERFORMANCE
    # --------------------------------------------------------

    st.markdown("### Category Performance")

    category_analysis = (
        filtered_df
        .groupby("Category")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Sales", "count"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
    )

    category_analysis["Profit Margin"] = np.where(
        category_analysis["Sales"] > 0,
        (
            category_analysis["Profit"] /
            category_analysis["Sales"]
        ) * 100,
        0
    )

    st.dataframe(
        category_analysis.style.format({
            "Sales": "${:,.2f}",
            "Profit": "${:,.2f}",
            "Orders": "{:,.0f}",
            "Quantity": "{:,.0f}",
            "Profit Margin": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CATEGORY SALES / PROFIT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            category_analysis,
            x="Category",
            y="Sales",
            title="Sales by Category",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            category_analysis,
            x="Category",
            y="Profit",
            title="Profit by Category",
            template="plotly_white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # REGION ANALYSIS
    # --------------------------------------------------------

    st.markdown("### 🌎 Regional Performance")

    region_analysis = (
        filtered_df
        .groupby("Region")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Orders=("Sales", "count")
        )
        .reset_index()
    )

    region_analysis["Margin"] = np.where(
        region_analysis["Sales"] > 0,
        region_analysis["Profit"] /
        region_analysis["Sales"] *
        100,
        0
    )

    fig = px.scatter(
        region_analysis,
        x="Sales",
        y="Profit",
        size="Orders",
        color="Region",
        hover_name="Region",
        title="Sales vs Profit by Region",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

    st.markdown("### 🏆 Top Products")

    product_analysis = (
        filtered_df
        .groupby("Product")
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Quantity=("Quantity", "sum")
        )
        .reset_index()
        .sort_values(
            "Sales",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        product_analysis.sort_values("Sales"),
        x="Sales",
        y="Product",
        orientation="h",
        text_auto=".2s",
        title="Top 10 Products by Sales",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DATA EXPLORER PAGE
# ============================================================

elif page == "Data Explorer":

    st.markdown("## 🔎 Data Explorer")

    st.write(
        f"Dataset contains **{len(filtered_df):,} rows** "
        f"and **{len(filtered_df.columns):,} columns**."
    )

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_text = st.text_input(
        "Search the dataset",
        placeholder="Search product, category, region..."
    )

    display_df = filtered_df.copy()

    if search_text:

        mask = display_df.astype(
            str
        ).apply(
            lambda row: row.str.contains(
                search_text,
                case=False,
                na=False
            ).any(),
            axis=1
        )

        display_df = display_df[mask]

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        hide_index=True
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv_data = display_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Data",
        data=csv_data,
        file_name="insight_spark_filtered_data.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    st.markdown("### 📋 Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            f"{len(display_df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            f"{len(display_df.columns):,}"
        )

    with col3:

        missing_values = (
            display_df.isna()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            f"{missing_values:,}"
        )

    st.markdown("### Column Information")

    info_df = pd.DataFrame({
        "Column": display_df.columns,
        "Data Type": [
            str(dtype)
            for dtype in display_df.dtypes
        ],
        "Missing Values": [
            display_df[col].isna().sum()
            for col in display_df.columns
        ],
        "Unique Values": [
            display_df[col].nunique()
            for col in display_df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AI INSIGHTS PAGE
# ============================================================

elif page == "AI Insights":

    st.markdown("## 🤖 AI Business Insights")

    st.write(
        "Automatically generated observations from the uploaded "
        "sales dataset."
    )

    insights = generate_insights(
        filtered_df
    )

    if insights:

        for i, insight in enumerate(
            insights,
            start=1
        ):

            st.markdown(
                f"""
                <div class="insight-box">
                    <div class="insight-title">
                        💡 Insight {i}
                    </div>

                    <div class="insight-text">
                        {insight}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "Upload data to generate insights."
        )

    # --------------------------------------------------------
    # QUICK ANALYSIS
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### ⚡ Quick Analysis"
    )

    question = st.text_input(
        "Ask a question about your data",
        placeholder=(
            "Example: Which category has the highest sales?"
        )
    )

    if st.button(
        "Analyze",
        type="primary"
    ):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            q = question.lower()

            # ------------------------------------------------
            # BEST CATEGORY
            # ------------------------------------------------

            if (
                "category" in q
                and (
                    "highest" in q
                    or "best" in q
                    or "top" in q
                )
            ):

                result = (
                    filtered_df
                    .groupby("Category")["Sales"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                if not result.empty:

                    best = result.index[0]

                    value = result.iloc[0]

                    st.success(
                        f"🏆 **{best}** has the highest "
                        f"sales with **{money(value)}**."
                    )

            # ------------------------------------------------
            # BEST REGION
            # ------------------------------------------------

            elif (
                "region" in q
                and (
                    "highest" in q
                    or "best" in q
                    or "top" in q
                )
            ):

                result = (
                    filtered_df
                    .groupby("Region")["Sales"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                if not result.empty:

                    best = result.index[0]

                    value = result.iloc[0]

                    st.success(
                        f"🌎 **{best}** is the top region "
                        f"with **{money(value)}** in sales."
                    )

            # ------------------------------------------------
            # PROFIT
            # ------------------------------------------------

            elif "profit" in q:

                st.success(
                    f"💰 Total profit is "
                    f"**{money(total_profit)}** "
                    f"with a profit margin of "
                    f"**{profit_margin:.1f}%**."
                )

            # ------------------------------------------------
            # SALES
            # ------------------------------------------------

            elif "sales" in q or "revenue" in q:

                st.success(
                    f"📊 Total sales are "
                    f"**{money(total_sales)}**."
                )

            # ------------------------------------------------
            # ORDERS
            # ------------------------------------------------

            elif "order" in q or "transaction" in q:

                st.success(
                    f"🛒 There are "
                    f"**{total_orders:,} transactions** "
                    f"in the selected dataset."
                )

            # ------------------------------------------------
            # AVERAGE
            # ------------------------------------------------

            elif (
                "average" in q
                or "aov" in q
            ):

                st.success(
                    f"📌 The average order value is "
                    f"**{money(average_order_value)}**."
                )

            # ------------------------------------------------
            # DEFAULT
            # ------------------------------------------------

            else:

                st.info(
                    "I can currently answer questions about "
                    "sales, profit, orders, categories, regions, "
                    "and average order value."
                )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### 🏆 Top 5 Products"
    )

    top_products = (
        filtered_df
        .groupby("Product")["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(5)
        .reset_index()
    )

    if not top_products.empty:

        for index, row in top_products.iterrows():

            st.markdown(
                f"""
                **{index + 1}. {row["Product"]}**
                — {money(row["Sales"])}
                """
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;padding:20px;">
        ✨ Insight Spark &nbsp; | &nbsp;
        AI-Powered Data Analytics Dashboard
        <br>
        Built with Python, Streamlit, Pandas & Plotly
    </div>
    """,
    unsafe_allow_html=True
)