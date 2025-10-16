import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

# --- App Configuration ---
st.set_page_config(page_title="Premium Data Analysis Portal", layout="wide")

# 🌌 --- Advanced Gradient Dark Theme & Responsive Styling ---
st.markdown("""
<style>
/* --- Global Dark Gradient Background --- */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #00fff7;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    transition: all 0.3s ease;
}

/* --- Sidebar Gradient --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1c3c, #2b2b5c);
    color: #00ffe7;
    border-radius: 15px;
    padding: 10px;
}

/* Sidebar text & labels */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #00ffe7 !important;
    font-weight: 600;
}

/* --- Gradient Headers --- */
h1, h2, h3, h4, h5, h6 {
    background: linear-gradient(90deg, #ff00c8, #00f0ff, #00ff94);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    transition: all 0.3s ease;
}

/* --- Buttons Gradient --- */
button[kind="primary"] {
    background: linear-gradient(135deg, #ff6ec4, #7873f5, #00f0ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    transition: transform 0.2s ease;
}
button[kind="primary"]:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #00f0ff, #7873f5, #ff6ec4) !important;
}

/* --- Dataframe headers & cells --- */
.stDataFrame thead {
    background: linear-gradient(90deg, #ff6ec4, #00f0ff, #00ff94) !important;
    color: #000 !important;
    font-weight: bold;
}
.stDataFrame tbody tr:nth-child(even) {
    background-color: rgba(255,255,255,0.05) !important;
}
.stDataFrame tbody tr:nth-child(odd) {
    background-color: rgba(255,255,255,0.02) !important;
}
.stDataFrame tbody td {
    color: #b3e5fc !important;
}

/* --- Alerts / Info boxes --- */
div[data-testid="stAlert"] {
    background: linear-gradient(90deg, #ff6ec4, #00f0ff, #00ff94);
    color: #000;
    border-left: 4px solid #fff;
    border-radius: 8px;
    padding: 10px;
}

/* --- Inputs / Textareas / Selects --- */
input, textarea, select {
    background: rgba(0,0,0,0.4) !important;
    color: #00ffe7 !important;
    border: 1px solid #00f0ff !important;
    border-radius: 8px !important;
}

/* --- Hover effects --- */
div[role="radiogroup"] label:hover,
div[role="combobox"] div:hover {
    background: rgba(0, 255, 255, 0.2) !important;
    border-radius: 5px;
}

/* --- Plotly charts responsive background --- */
.js-plotly-plot {
    background: transparent !important;
}

/* --- Smooth scrolling and responsive spacing --- */
.css-18e3th9 {
    padding: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📊 Premium Data Analysis Portal")


# --- Your original code below (untouched functionality) ---
page = st.sidebar.radio(
    "Select Step",
    [
        "Upload Dataset",
        "EDA & Summary",
        "Handle Missing Values",
        "Validate Missing Value Handling",
        "Compare Numeric Columns",
        "Visualizations",
        "Download Data"
    ]
)

if "df" not in st.session_state:
    st.session_state.df = None

# Step 1: Upload Dataset
if page == "Upload Dataset":
    st.header("1️⃣ Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")
        st.session_state.original_missing = st.session_state.df.isnull().sum().to_frame(name="Missing Values")
        st.dataframe(st.session_state.df)

# (Continue all the rest of your original code here unchanged: EDA, Missing Value Handling, Compare, Visualizations, Download)


# --- Step 2: EDA & Summary ---
elif page == "EDA & Summary":
    st.header("2️⃣ Exploratory Data Analysis & Summary")

    df = st.session_state.df
    if df is not None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

        # --- Search / Filter Section ---
        st.subheader("🔍 Search / Filter Data Before Editing")
        col1, col2, col3 = st.columns([3, 3, 1.5])
        with col1:
            search_col = st.selectbox("Select column to search", df.columns)
        with col2:
            search_text = st.text_input(f"Search text in '{search_col}'")
        with col3:
            case_sensitive = st.checkbox("Case Sensitive Search", value=False)

        if search_text:
            if case_sensitive:
                filtered_df = df[df[search_col].astype(str).str.contains(search_text, regex=False)]
            else:
                filtered_df = df[df[search_col].astype(str).str.contains(search_text, case=False, regex=False)]
        else:
            filtered_df = df.copy()

        st.dataframe(filtered_df, use_container_width=True)

        # --- Editable Table Section ---
        st.subheader("✏️ Edit Data")
        st.caption("You can edit cells directly below. Changes will be applied to the main dataset.")
        edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True)
        st.session_state.df.update(edited_df)

        # --- Numeric Summary with Total Row ---
        st.subheader("📈 Numeric Summary")
        if numeric_cols:
            summary = st.session_state.df[numeric_cols].describe().T
            summary.loc["Total"] = st.session_state.df[numeric_cols].sum()
            summary = summary.round(2)
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No numeric columns found.")

        # --- Categorical Summary ---
        st.subheader("📝 Categorical Summary")
        for col in categorical_cols:
            with st.expander(f"📊 Distribution of '{col}'"):
                st.write(st.session_state.df[col].value_counts())

        # --- Missing Values Summary ---
        st.subheader("⚠️ Missing Values Summary")
        missing_df = st.session_state.df.isnull().sum().to_frame(name="Missing Values")
        missing_df["% Missing"] = (missing_df["Missing Values"] / len(st.session_state.df)) * 100

        # Add total missing row
        total_missing = pd.DataFrame({
            "Missing Values": [missing_df["Missing Values"].sum()],
            "% Missing": [missing_df["Missing Values"].sum() / len(st.session_state.df) * 100]
        }, index=["Total"])

        missing_df = pd.concat([missing_df, total_missing])
        st.dataframe(missing_df.style.format({"% Missing": "{:.2f}"}), use_container_width=True)

    else:
        st.info("Please upload a dataset first.")

# --- Step 3: Handle Missing Values ---
elif page == "Handle Missing Values":
    st.header("3️⃣ Handle Missing Values")
    df = st.session_state.df
    if df is not None:
        st.session_state.original_missing = df.isnull().sum().to_frame(name="Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            cols_to_handle = st.multiselect("Select columns to handle missing values", missing_cols)
            if cols_to_handle:
                options = st.multiselect(
                    "Choose methods to apply",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with constant"]
                )
                constant_value = None
                if "Fill with constant" in options:
                    constant_value = st.text_input("Enter constant value for filling")

                if st.button("Apply to selected columns"):
                    for col in cols_to_handle:
                        for option in options:
                            if option == "Drop rows":
                                df = df.dropna(subset=[col])
                            elif option == "Fill with mean":
                                if df[col].dtype in ['int64', 'float64']:
                                    df[col] = df[col].fillna(df[col].mean())
                            elif option == "Fill with median":
                                if df[col].dtype in ['int64', 'float64']:
                                    df[col] = df[col].fillna(df[col].median())
                            elif option == "Fill with mode":
                                df[col] = df[col].fillna(df[col].mode()[0])
                            elif option == "Fill with constant" and constant_value:
                                df[col] = df[col].fillna(constant_value)
                    st.session_state.df = df
                    st.success("✅ Missing values handled successfully!")
                    st.dataframe(df)
        else:
            st.info("No missing values found in the dataset.")
    else:
        st.info("Please upload a dataset first.")

# --- Step 4: Validate Missing Value Handling ---
elif page == "Validate Missing Value Handling":
    st.header("4️⃣ Validate Missing Value Handling")

    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df

        # Recalculate missing values
        missing_df = df.isnull().sum().to_frame(name="Missing Values")
        missing_df["% Missing"] = (missing_df["Missing Values"] / len(df)) * 100
        total_missing = missing_df["Missing Values"].sum()

        st.subheader("🔎 Missing Value Summary (After Handling)")
        st.dataframe(missing_df.style.format({"% Missing": "{:.2f}"}), use_container_width=True)

        # Validation Message
        if total_missing == 0:
            st.success("✅ All missing values handled successfully! 🎉")
            st.balloons()
        else:
            st.warning(f"⚠️ {total_missing} missing values still remain in the dataset.")
            st.info("You may revisit 'Handle Missing Values' to clean them further.")

        # Before vs After Comparison
        st.subheader("📊 Missing Values Comparison (Before vs After)")
        if "original_missing" in st.session_state:
            before = st.session_state.original_missing
            after = missing_df
            compare_df = pd.DataFrame({
                "Before": before["Missing Values"],
                "After": after["Missing Values"]
            }).fillna(0)

            fig = px.bar(
                compare_df,
                barmode="group",
                title="Missing Values: Before vs After Handling",
                labels={"index": "Columns", "value": "Missing Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Original missing value snapshot not found. Try re-uploading and cleaning again.")

    else:
        st.info("Please upload and handle missing values first.")

# --- Step 5: Compare Numeric Columns (Professional + Auto Insights + Outlier Handling) ---
elif page == "Compare Numeric Columns":
    st.header("5️⃣ Compare Numeric Columns | Professional Analysis with Auto Insights")
    df = st.session_state.df

    if df is not None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
        date_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()

        if len(numeric_cols) < 2:
            st.info("At least 2 numeric columns are required for comparison.")
        else:
            # --- Column Selection ---
            compare_cols = st.multiselect(
                "Select numeric columns to compare",
                numeric_cols,
                default=numeric_cols[:2],
                help="Pick 2 or more numeric columns to analyze relationships."
            )

            # Optional grouping
            group_col = st.selectbox(
                "Optional: Group data by categorical column",
                [None] + categorical_cols,
                index=0
            )

            # Time series mode
            time_series_mode = False
            if date_cols:
                time_series_mode = st.checkbox("Enable Time-Series Mode (use Date column for X-axis)")
                if time_series_mode:
                    date_col = st.selectbox("Select Date Column", date_cols, index=0)
                    df_sorted = df.sort_values(by=date_col)
                else:
                    df_sorted = df.copy()
            else:
                df_sorted = df.copy()

            # --- Chart Selection ---
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line", "Scatter", "Scatter Matrix", "Box/Violin", "Bar Chart", "Histogram/Density", "3D Scatter", "Correlation Heatmap"]
            )

            # --- Plotting ---
            if compare_cols:
                fig = None

                # Line Chart
                if chart_type == "Line":
                    x_col = date_col if time_series_mode else df_sorted.index
                    fig = px.line(
                        df_sorted,
                        x=x_col,
                        y=compare_cols,
                        color=group_col,
                        markers=True,
                        title="📈 Line Chart of Numeric Columns"
                    )

                # Scatter Plot
                elif chart_type == "Scatter":
                    if len(compare_cols) == 2:
                        fig = px.scatter(
                            df_sorted,
                            x=compare_cols[0],
                            y=compare_cols[1],
                            color=group_col,
                            trendline="ols",
                            title=f"🔁 Scatter Plot: {compare_cols[0]} vs {compare_cols[1]}"
                        )
                    else:
                        st.warning("Select exactly 2 numeric columns for scatter plot.")

                # Scatter Matrix
                elif chart_type == "Scatter Matrix":
                    fig = px.scatter_matrix(
                        df_sorted,
                        dimensions=compare_cols,
                        color=group_col,
                        title="🔀 Pairwise Scatter Matrix"
                    )
                    fig.update_traces(diagonal_visible=False)

                # Box/Violin
                elif chart_type == "Box/Violin":
                    fig = px.box(
                        df_sorted,
                        y=compare_cols[0],
                        x=group_col if group_col else None,
                        points="all",
                        title=f"📊 Box Plot / Violin Plot for {compare_cols[0]}"
                    )

                # Bar Chart
                elif chart_type == "Bar Chart":
                    if group_col:
                        df_grouped = df_sorted.groupby(group_col)[compare_cols].mean().reset_index()
                        fig = px.bar(
                            df_grouped,
                            x=group_col,
                            y=compare_cols,
                            barmode='group',
                            title="📊 Grouped Bar Chart (Mean Values)"
                        )
                    else:
                        st.warning("Select a categorical column to group by for bar chart.")

                # Histogram / Density
                elif chart_type == "Histogram/Density":
                    col = compare_cols[0]
                    fig = ff.create_distplot(
                        [df_sorted[col].dropna()],
                        group_labels=[col],
                        show_hist=True,
                        show_rug=False
                    )

                # 3D Scatter
                elif chart_type == "3D Scatter":
                    if len(compare_cols) >= 3:
                        fig = px.scatter_3d(
                            df_sorted,
                            x=compare_cols[0],
                            y=compare_cols[1],
                            z=compare_cols[2],
                            color=group_col,
                            title="🌐 3D Scatter Plot"
                        )
                    else:
                        st.warning("Select at least 3 numeric columns for 3D scatter plot.")

                # Correlation Heatmap
                elif chart_type == "Correlation Heatmap":
                    corr = df_sorted[compare_cols].corr()
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        title="🔥 Correlation Heatmap"
                    )

                # Render Plot
                if fig:
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)

                # --- Auto Insights ---
                st.subheader("💡 Auto Insights")
                insights = []
                # Highlight correlations
                corr = df_sorted[compare_cols].corr()
                for i in corr.columns:
                    for j in corr.columns:
                        if i != j:
                            val = corr.loc[i, j]
                            if abs(val) >= 0.7:
                                insights.append(f"Strong correlation: {i} ↔ {j} (r={val:.2f})")
                            elif abs(val) >= 0.4:
                                insights.append(f"Moderate correlation: {i} ↔ {j} (r={val:.2f})")

                # Highlight outliers
                for col in compare_cols:
                    q1 = df_sorted[col].quantile(0.25)
                    q3 = df_sorted[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = df_sorted[(df_sorted[col] < q1 - 1.5*iqr) | (df_sorted[col] > q3 + 1.5*iqr)][col]
                    if len(outliers) > 0:
                        insights.append(f"Column {col} has {len(outliers)} potential outliers.")

                if insights:
                    for insight in insights:
                        st.info(insight)
                else:
                    st.success("No strong correlations or outliers detected.")

                # --- NEW: Outlier Summary Table ---
                st.subheader("📊 Outlier Summary & Treatment Suggestions")
                outlier_data = []
                for col in compare_cols:
                    q1 = df_sorted[col].quantile(0.25)
                    q3 = df_sorted[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = df_sorted[(df_sorted[col] < lower) | (df_sorted[col] > upper)][col]
                    percentage = (len(outliers) / len(df_sorted)) * 100

                    if len(outliers) == 0:
                        suggestion = "✅ No outliers detected."
                    elif percentage < 3:
                        suggestion = "⚠️ Minor outliers — review data entries."
                    elif percentage < 10:
                        suggestion = "🚨 Moderate outliers — consider capping or transformation."
                    else:
                        suggestion = "🔥 High outliers — investigate data distribution."

                    outlier_data.append({
                        "Column": col,
                        "Outliers Detected": len(outliers),
                        "% of Data": f"{percentage:.2f}%",
                        "Suggested Action": suggestion
                    })

                outlier_df = pd.DataFrame(outlier_data)
                st.dataframe(outlier_df, use_container_width=True)

                st.markdown(
                    "💡 **Insight Tip:** Outliers can distort averages and correlation. "
                    "You can apply capping, trimming, or log transformation to normalize your data."
                )

            else:
                st.warning("Select at least 2 numeric columns for comparison.")

    else:
        st.info("Please upload a dataset first.")


# --- Step 6: Visualizations (Enhanced) ---
elif page == "Visualizations":
    st.header("6️⃣ Data Visualization Studio 🎨")
    df = st.session_state.df

    if df is not None:
        st.markdown("""
        Unlock insights from your dataset with interactive visualizations.  
        Choose different chart types to explore trends, distributions, and relationships between variables.
        """)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(exclude='number').columns.tolist()

        # --- Column Selection ---
        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            chart_x = st.selectbox("Select X-axis", df.columns, key="vis_x")
        with col2:
            chart_y = st.selectbox("Select Y-axis (optional)", [None] + list(df.columns), key="vis_y")
        with col3:
            chart_type = st.selectbox(
                "Select chart type",
                [
                    "Scatter", "Bar", "Line", "Box", "Histogram",
                    "Pie", "Density", "Correlation Heatmap", "Violin", "3D Scatter"
                ],
                key="chart_type"
            )

        # --- Additional Controls ---
        with st.expander("⚙️ Advanced Visualization Settings"):
            color_col = st.selectbox("Add Color Group (optional)", [None] + categorical_cols)
            size_col = st.selectbox("Bubble Size (for Scatter/3D)", [None] + numeric_cols)
            log_scale = st.checkbox("Use Logarithmic Scale (Y-axis)", value=False)
            show_trendline = st.checkbox("Add Regression Trendline (Scatter only)", value=False)

        # --- Plot Section ---
        if chart_type == "Scatter":
            fig = px.scatter(
                df, x=chart_x, y=chart_y,
                color=color_col, size=size_col,
                trendline="ols" if show_trendline else None,
                title=f"Scatter Plot: {chart_x} vs {chart_y}",
                template="plotly_white"
            )

        elif chart_type == "3D Scatter":
            numeric_for_3d = st.multiselect(
                "Select 3 numeric columns for 3D scatter (X, Y, Z)", numeric_cols, default=numeric_cols[:3]
            )
            if len(numeric_for_3d) == 3:
                fig = px.scatter_3d(
                    df, x=numeric_for_3d[0], y=numeric_for_3d[1], z=numeric_for_3d[2],
                    color=color_col, size=size_col, title="3D Scatter Visualization"
                )
            else:
                st.warning("Please select exactly 3 numeric columns for 3D scatter plot.")

        elif chart_type == "Bar":
            fig = px.bar(df, x=chart_x, y=chart_y, color=color_col, barmode='group')

        elif chart_type == "Line":
            fig = px.line(df, x=chart_x, y=chart_y, color=color_col, markers=True)

        elif chart_type == "Box":
            fig = px.box(df, x=chart_x, y=chart_y, color=color_col, points="all")

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=chart_x, color=color_col, nbins=30, marginal="box")

        elif chart_type == "Pie":
            if chart_y and df[chart_y].dtype in ['int64', 'float64']:
                fig = px.pie(df, names=chart_x, values=chart_y)
            else:
                pie_counts = df[chart_x].value_counts().reset_index()
                fig = px.pie(pie_counts, names='index', values=chart_x, title="Category Distribution")

        elif chart_type == "Density":
            if chart_y in numeric_cols:
                fig = ff.create_distplot([df[chart_y].dropna()], [chart_y], show_hist=True, show_rug=False)
            else:
                st.warning("Please select a numeric column for density plot.")
                fig = None

        elif chart_type == "Correlation Heatmap":
            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                title="🔥 Correlation Heatmap (Numeric Features)"
            )
            st.markdown("""
            **How to read this heatmap:**  
            - 🔵 Negative correlation → When one increases, the other decreases  
            - 🟢 Positive correlation → Both increase together  
            - ⚪ Close to zero → Weak or no relationship
            """)

        elif chart_type == "Violin":
            fig = px.violin(df, x=chart_x, y=chart_y, color=color_col, box=True, points="all")

        # --- Apply Log Scale if needed ---
        if fig and log_scale:
            fig.update_yaxes(type="log")

        # --- Render Chart ---
        if fig:
            fig.update_layout(template="plotly_dark", title_x=0.4)
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Visualization generated successfully!")
        else:
            st.warning("Please select valid columns for the selected chart type.")

    else:
        st.info("Please upload a dataset first to start visualizing.")


# --- Step 7: Download Data ---
elif page == "Download Data":
    st.header("7️⃣ Download Updated Dataset")
    df = st.session_state.df
    if df is not None:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download CSV", data=csv, file_name="updated_dataset.csv", mime="text/csv")
    else:
        st.info("Please upload a dataset first.")
