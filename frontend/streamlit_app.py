import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Exoplanet Habitability", layout="wide")

DEFAULT_API = "https://habitability-of-exoplanets-wpdp.onrender.com"

st.sidebar.title("Settings")
BASE_URL = st.sidebar.text_input("API base URL", value=DEFAULT_API)

st.title("Exoplanet Habitability Explorer 🔭")

tabs = st.tabs(["Predict", "Top Habitable", "Feature Importance", "Model & Sampling Comparisons", "Dataset"])

final_features = [
    "HSI",
    "planet_density",
    "pl_eqt",
    "pl_rade",
    "pl_bmasse",
    "st_teff",
    "star_luminosity",
    # star types are encoded as one-hot
    "star_type_M",
    "star_type_K",
    "star_type_G",
]

# --- Predict Tab ---
with tabs[0]:
    st.header("Predict Habitability")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            HSI = st.number_input("HSI", value=0.5, step=0.01, format="%.3f")
            planet_density = st.number_input("Planet Density", value=5.5, step=0.01)
            pl_eqt = st.number_input("Equilibrium Temp (K)", value=300.0, step=1.0)
        with col2:
            pl_rade = st.number_input("Radius (Earth radii)", value=1.0, step=0.01)
            pl_bmasse = st.number_input("Mass (Earth masses)", value=1.0, step=0.01)
            st_teff = st.number_input("Star Teff (K)", value=5800.0, step=1.0)
        with col3:
            star_luminosity = st.number_input("Star Luminosity (L☉)", value=1.0, step=0.01)
            star_type = st.selectbox("Star Type", options=["G", "K", "M"], index=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "HSI": float(HSI),
            "planet_density": float(planet_density),
            "pl_eqt": float(pl_eqt),
            "pl_rade": float(pl_rade),
            "pl_bmasse": float(pl_bmasse),
            "st_teff": float(st_teff),
            "star_luminosity": float(star_luminosity),
            "star_type_M": 1 if star_type == "M" else 0,
            "star_type_K": 1 if star_type == "K" else 0,
            "star_type_G": 1 if star_type == "G" else 0,
        }
        try:
            res = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
            prob = data.get("habitability_probability")
            label = data.get("habitability_prediction")
            filled = data.get("filled_defaults", [])
            st.success(f"Prediction: {label} — Probability: {prob}")
            if filled:
                st.info(f"Missing inputs were filled with dataset medians for: {', '.join(filled)}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# --- Top Habitable Tab ---
with tabs[1]:
    st.header("Top Habitable Exoplanets")
    try:
        # Load data from CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "../notebook/exo_habitability_final.csv")
        df = pd.read_csv(csv_path)
        
        # Display data table
        st.subheader("Top Habitable Exoplanets Data")
        top_df = df.sort_values("HSI", ascending=False).head(20)
        #st.dataframe(top_df, use_container_width=True)
        st.dataframe(df, width="stretch")

        # Export functionality
        st.subheader("Export Reports")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Export to Excel
            if st.button("📊 Export as Excel", key="export_excel"):
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Top 20 planets sheet
                    top_df.to_excel(writer, sheet_name='Top 20 Planets', index=False)
                    
                    # Summary statistics sheet
                    summary_data = {
                        'Metric': [
                            'Total Exoplanets',
                            'Habitable Planets (P > 0.5)',
                            'Avg Habitability Probability',
                            'Max HSI Score',
                            'Avg HSI Score',
                            'Report Generated'
                        ],
                        'Value': [
                            len(df),
                            (df["habitability_probability"] > 0.5).sum(),
                            f"{df['habitability_probability'].mean():.4f}",
                            f"{df['HSI'].max():.4f}",
                            f"{df['HSI'].mean():.4f}",
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                buffer.seek(0)
                st.download_button(
                    label="💾 Download Excel File",
                    data=buffer,
                    file_name=f"exoplanet_habitability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Excel file ready for download!")
        
        with export_col2:
            # Export CSV
            if st.button("📄 Export as CSV", key="export_csv"):
                csv_buffer = top_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_buffer,
                    file_name=f"top_habitable_exoplanets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success("✅ CSV file ready for download!")
        
        with export_col3:
            # Export as HTML
            if st.button("🌐 Export as HTML", key="export_html"):
                html_content = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        h2 {{ color: #34495e; margin-top: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #bdc3c7; padding: 12px; text-align: left; }}
                        th {{ background-color: #3498db; color: white; }}
                        tr:nth-child(even) {{ background-color: #ecf0f1; }}
                        .summary {{ background-color: #d5f4e6; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 12px; }}
                    </style>
                </head>
                <body>
                    <h1>🔭 Exoplanet Habitability Report</h1>
                    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <div class="summary">
                        <h2>Summary Statistics</h2>
                        <p><strong>Total Exoplanets:</strong> {len(df)}</p>
                        <p><strong>Habitable Planets (P > 0.5):</strong> {(df["habitability_probability"] > 0.5).sum()}</p>
                        <p><strong>Average Habitability Probability:</strong> {df['habitability_probability'].mean():.4f}</p>
                        <p><strong>Maximum HSI Score:</strong> {df['HSI'].max():.4f}</p>
                        <p><strong>Average HSI Score:</strong> {df['HSI'].mean():.4f}</p>
                    </div>
                    
                    <h2>Top 20 Habitable Exoplanets</h2>
                    {top_df.to_html(index=False, border=0)}
                    
                    <div class="footer">
                        <p>This report was generated by the Exoplanet Habitability Explorer</p>
                    </div>
                </body>
                </html>
                """
                st.download_button(
                    label="💾 Download HTML File",
                    data=html_content,
                    file_name=f"exoplanet_habitability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                st.success("✅ HTML file ready for download!")
        
        st.divider()
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of top 20 planets by HSI
            if not df.empty and "HSI" in df.columns:
                fig_hsi = px.bar(
                    top_df,
                    x="pl_name",
                    y="HSI",
                    title="Top 20 Planets by Habitability Score Index (HSI)",
                    labels={"pl_name": "Planet Name", "HSI": "HSI"},
                    color="HSI",
                    color_continuous_scale="RdYlGn"
                )
                fig_hsi.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig_hsi, use_container_width=True)
        
        with col2:
            # Habitability Score Distribution
            if "habitability_probability" in df.columns:
                fig_dist = px.histogram(
                    df,
                    x="habitability_probability",
                    nbins=30,
                    title="Distribution of Habitability Probabilities",
                    labels={"habitability_probability": "Habitability Probability"},
                    color_discrete_sequence=["#3498db"]
                )
                fig_dist.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Additional statistics
        st.subheader("Habitability Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            habitable_count = (df["habitability_probability"] > 0.2).sum()
            st.metric("Habitable Planets (P > 0.2)", habitable_count)
        
        with col2:
            avg_probability = df["habitability_probability"].mean()
            st.metric("Avg Habitability Probability", f"{avg_probability:.4f}")
        
        with col3:
            max_hsi = df["HSI"].max()
            st.metric("Max HSI Score", f"{max_hsi:.4f}")
        
        with col4:
            avg_hsi = df["HSI"].mean()
            st.metric("Avg HSI Score", f"{avg_hsi:.4f}")
            
    except FileNotFoundError:
        st.error("Exoplanet data CSV file not found. Please ensure 'exo_habitability_final.csv' exists in the notebook folder.")
    except Exception as e:
        st.error(f"Failed to load top habitable data: {e}")

# --- Feature Importance ---
with tabs[2]:
    st.header("Feature Importance")
    try:
        # Load feature importance data from CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "../notebook/feature_importance_ranking.csv")
        fi = pd.read_csv(csv_path)
        
        # Rename 'Feature' column to 'feature' for consistency
        fi = fi.rename(columns={"Feature": "feature"})
        
        # Calculate average importance for visualization
        importance_cols = ["LR_Coeff", "RF_Importance", "XGB_Importance", "Perm_Importance", "Average_Importance"]
        
        # Display data table
        st.subheader("Feature Importance Rankings")
        st.dataframe(fi, use_container_width=True)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart with average importance
            fig_avg = px.bar(
                fi.sort_values("Average_Importance", ascending=True), 
                y="feature", 
                x="Average_Importance", 
                orientation="h",
                title="Average Feature Importance",
                labels={"feature": "Feature", "Average_Importance": "Average Importance"},
                color="Average_Importance",
                color_continuous_scale="Viridis"
            )
            fig_avg.update_layout(height=600)
            #st.plotly_chart(fig_avg, use_container_width=True)
            st.plotly_chart(fig, width="stretch")

        
        with col2:
            # Comparison of different importance methods
            importance_methods = ["LR_Coeff", "RF_Importance", "XGB_Importance", "Perm_Importance"]
            fi_melted = fi.melt(id_vars="feature", value_vars=importance_methods, 
                                var_name="Method", value_name="Importance")
            fig_compare = px.bar(
                fi_melted,
                x="feature",
                y="Importance",
                color="Method",
                title="Feature Importance by Method",
                barmode="group"
            )
            fig_compare.update_layout(height=600, xaxis_tickangle=-45)
            st.plotly_chart(fig_compare, use_container_width=True)
        
    except FileNotFoundError:
        st.error("Feature importance CSV file not found. Please ensure 'feature_importance_ranking.csv' exists in the notebook folder.")
    except Exception as e:
        st.error(f"Failed to load feature importance: {e}")

# --- Model & Sampling Comparisons ---
with tabs[3]:
    st.header("Model & Sampling Comparisons")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    col1, col2 = st.columns(2)

    # -------- Model Comparison --------
    with col1:
        st.subheader("Model Comparison")
        try:
            model_csv_path = os.path.join(
                current_dir, "../notebook/baseline_models_comparison.csv"
            )

            if not os.path.exists(model_csv_path):
                st.error("baseline_models_comparison.csv not found in notebook folder")
            else:
                mc = pd.read_csv(model_csv_path)
                st.dataframe(mc, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load model comparisons: {e}")

    # -------- Sampling Techniques --------
    with col2:
        st.subheader("Sampling Techniques")
        try:
            sampling_csv_path = os.path.join(
                current_dir, "../notebook/sampling_techniques_comparison.csv"
            )

            if not os.path.exists(sampling_csv_path):
                st.error("sampling_techniques_comparison.csv not found in notebook folder")
            else:
                sc = pd.read_csv(sampling_csv_path)
                st.dataframe(sc, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load sampling comparisons: {e}")

    
    # Star-Planet Parameter Correlations
    st.subheader("Star-Planet Parameter Correlations")
    try:
        # Load exoplanet data for correlations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exo_path = os.path.join(current_dir, "../notebook/exo_habitability_final.csv")
        exo_df = pd.read_csv(exo_path)
        
        # Select star and planet parameters
        correlation_features = [
            "HSI", "habitability_probability", "pl_rade", "pl_bmasse", "pl_eqt", 
            "planet_density", "pl_insol", "st_teff", "star_luminosity", "pl_orbper"
        ]
        
        # Filter to available columns
        available_cols = [col for col in correlation_features if col in exo_df.columns]
        corr_matrix = exo_df[available_cols].corr()
        
        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        fig_corr.update_layout(
            title="Star-Planet Parameter Correlations",
            xaxis_tickangle=-45,
            height=700,
            width=900
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not load correlation plot: {e}")

# --- Full dataset / extras ---
with tabs[4]:
    st.header("Exoplanet Dataset")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "../notebook/exo_habitability_final.csv")

        if not os.path.exists(csv_path):
            st.error("Dataset file not found: notebook/exo_habitability_final.csv")
        else:
            exo = pd.read_csv(csv_path)
            st.dataframe(exo, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")


st.markdown("---")
st.caption("Tip: run the Flask API (`python notebook/api/app.py`) and then run this Streamlit app (`streamlit run frontend/streamlit_app.py`).")
