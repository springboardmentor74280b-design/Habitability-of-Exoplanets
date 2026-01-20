import matplotlib
matplotlib.use('Agg') # Fix for server-side plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from model_utils import apply_physics_engine, load_model

def get_plot_as_base64():
    """Helper to convert current plot to base64 string"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_dashboard_plots():
    """Generates all 5 Global Dashboard plots."""
    print("🎨 Generating Global Dashboard Plots...")
    
    # 1. Load Global Data
    try:
        df = pd.read_csv('phl_exoplanet_catalog.csv')
        df = apply_physics_engine(df) # Clean & Physics check
        pipeline = load_model()
    except Exception as e:
        print(f"Error loading data for dashboard: {e}")
        return {}

    # Features we care about
    features = ['P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
                'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 
                'S_RADIUS', 'S_MASS', 'S_METALLICITY']
    
    # Clean Data for Plotting
    plot_df = df[features].dropna()
    X = plot_df
    y_pred = pipeline.predict(X) 
    
    plots = {}

    # --- PLOT 1: Feature Importance (ROBUST FIX) ---
    plt.figure(figsize=(10, 6))
    
    # FIX: Automatically get the last step (the model) regardless of its name
    try:
        # specific_model = pipeline.named_steps['classifier'] # Old crashing way
        specific_model = pipeline.steps[-1][1] # New robust way
        importances = specific_model.feature_importances_
        
        sns.barplot(x=importances, y=features, palette='viridis')
        plt.title('AI Scan: Which Physics Laws Matter Most?')
        plt.xlabel('Importance Score')
    except Exception as e:
        print(f"⚠️ Could not generate feature importance: {e}")
        plt.text(0.5, 0.5, "Feature Importance Not Available", ha='center')
        
    plots['feature_importance'] = get_plot_as_base64()
    plt.close()

# --- PLOT 2: Habitability Distribution (Pie Chart) ---
    plt.figure(figsize=(8, 6)) # Made figure wider for legend
    unique, counts = np.unique(y_pred, return_counts=True)
    
    # Safe map
    counts_map = {0:0, 1:0, 2:0}
    for u, c in zip(unique, counts):
        if u in counts_map:
            counts_map[u] = c
    
    # Calculate percentages for the legend
    total = sum(counts_map.values())
    pct_non = (counts_map[0] / total) * 100 if total > 0 else 0
    pct_hab = (counts_map[1] / total) * 100 if total > 0 else 0
    pct_opt = (counts_map[2] / total) * 100 if total > 0 else 0

    labels = [f'Non-Habitable ({pct_non:.1f}%)', 
              f'Habitable ({pct_hab:.1f}%)', 
              f'Optimistic ({pct_opt:.1f}%)']
    
    # Plot Pie Chart without labels inside
    plt.pie([counts_map[0], counts_map[1], counts_map[2]], 
            colors=['#bdc3c7', '#2ecc71', '#f1c40f'], startangle=140)
    
    plt.title('Global Exoplanet Habitability')
    
    # Add Legend Below
    plt.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=12)
    
    plots['distribution'] = get_plot_as_base64()
    plt.close()

    # --- PLOT 3: Correlation Matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(plot_df.corr(), annot=True, fmt=".1f", cmap='coolwarm', linewidths=0.5)
    plt.title('Physics Correlation Matrix')
    plots['correlation'] = get_plot_as_base64()
    plt.close()


    # --- PREPARE FOR PCA / t-SNE ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Limit data for speed (use first 500-1000 points)
    X_sample = X_scaled[:1000] 
    y_sample = y_pred[:1000]

    # DEFINING COLORS & LABELS
    # 0=Non-Habitable (Gray), 1=Habitable (Green), 2=Optimistic (Orange)
    label_map = {0: 'Non-Habitable', 1: 'Habitable', 2: 'Optimistic'}
    color_map = {0: '#7f8c8d', 1: '#2ecc71', 2: '#d35400'}
    
    # Define the Layer Order: Draw 0 first (bottom), then 2, then 1 (top)
    z_order = [0, 2, 1] 

    # --- PLOT 4: PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    
    # Loop ensures we draw layers in the correct order
    for cls in z_order:
        # Find all points belonging to this class
        indices = np.where(y_sample == cls)
        if len(indices[0]) > 0:
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], 
                        c=color_map[cls], label=label_map[cls],
                        alpha=0.8, s=30, edgecolors='w', linewidth=0.5)
    
    plt.title('PCA: The Shape of Habitability (2D)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='best') # Legend is now auto-generated from the loop labels
    
    plots['pca'] = get_plot_as_base64()
    plt.close()

    # --- PLOT 5: t-SNE ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))
    
    # Loop again for t-SNE layers
    for cls in z_order:
        indices = np.where(y_sample == cls)
        if len(indices[0]) > 0:
            plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                        c=color_map[cls], label=label_map[cls],
                        alpha=0.8, s=30, edgecolors='w', linewidth=0.5)
    
    plt.title('t-SNE: Hidden Clusters')
    plt.legend(loc='best')
    
    plots['tsne'] = get_plot_as_base64()
    plt.close()

    print("✅ Dashboard Plots Generated.")
    return plots

def generate_user_plots(df, limit_data=True):
    """
    Generates plots for a specific user-uploaded dataset.
    limit_data: If True, downsamples large datasets to 500 points for speed.
    """
    print(f"🎨 Generating User Plots (Limit: {limit_data})...")
    plots = {}
    
    # 1. Prepare Data
    map_pred = {'Non-Habitable': 0, 'Habitable': 1, 'Optimistic': 2}
    if 'prediction' in df.columns:
        y = df['prediction'].map(map_pred).fillna(0)
    else:
        y = np.zeros(len(df))
        
    features = ['P_MASS_EST', 'P_RADIUS_EST', 'P_PERIOD', 'P_DISTANCE', 
                'P_TEMP_EQUIL', 'S_TEMPERATURE', 'S_LUMINOSITY', 
                'S_RADIUS', 'S_MASS', 'S_METALLICITY']
    existing_feats = [f for f in features if f in df.columns]
    X = df[existing_feats].fillna(0)

    # --- NEW: PIE CHART (Clean Layout) ---
    plt.figure(figsize=(8, 6)) # Slightly wider for the legend
    counts = df['prediction'].value_counts()
    
    # 1. Strict Color Mapping (Fixes the Green/Gray issue)
    color_map = {'Habitable': '#2ecc71', 'Optimistic': '#d35400', 'Non-Habitable': '#95a5a6'}
    pie_colors = [color_map.get(label, '#95a5a6') for label in counts.index]
    
    # 2. Calculate Percentages for the Legend
    total = sum(counts)
    legend_labels = [f"{label}: {value/total*100:.1f}%" for label, value in zip(counts.index, counts)]
    
    # 3. Draw Pie (No labels inside)
    wedges, texts = plt.pie(counts, colors=pie_colors, startangle=140)
    
    # 4. Add Legend Below
    plt.legend(wedges, legend_labels,
               title="Verdict Distribution",
               loc="center",
               bbox_to_anchor=(0.5, -0.15), # Move below the chart
               frameon=False, # Clean look (no box border)
               ncol=1,
               fontsize=12,             # <--- Larger Text
               title_fontsize=13)       # <--- Larger Title 
               
    plt.title('Habitability Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout() # Ensures nothing gets cut off
    plots['pie'] = get_plot_as_base64()
    plt.close()


    # --- HANDLING DATA LIMITS FOR SCATTER PLOTS ---
    # PCA and t-SNE are slow with 4000+ points.
    if limit_data and len(df) > 800:
        # Keep ALL Habitable/Optimistic, but sample the Non-Habitable
        # This is a "Smart Sample" so we don't lose the important stuff
        mask_interesting = (y == 1) | (y == 2)
        mask_boring = (y == 0)
        
        X_interesting = X[mask_interesting]
        y_interesting = y[mask_interesting]
        
        # Take only 500 boring points
        X_boring = X[mask_boring].sample(n=min(500, sum(mask_boring)), random_state=42)
        y_boring = y[mask_boring].sample(n=min(500, sum(mask_boring)), random_state=42)
        
        X_final = pd.concat([X_interesting, X_boring])
        y_final = pd.concat([y_interesting, y_boring])
    else:
        # High Precision Mode: Use everything
        X_final = X
        y_final = y

    # --- PLOT 2: Correlation Matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_final.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plots['correlation'] = get_plot_as_base64()
    plt.close()

    # --- PREPARE PCA/t-SNE ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    label_map = {0: 'Non-Habitable', 1: 'Habitable', 2: 'Optimistic'}
    color_map = {0: '#7f8c8d', 1: '#2ecc71', 2: '#d35400'}
    z_order = [0, 2, 1]

    # --- PLOT 3: PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(6, 5))
    for cls in z_order:
        indices = np.where(y_final == cls)
        if len(indices[0]) > 0:
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], 
                        c=color_map[cls], label=label_map[cls],
                        alpha=0.8, s=40, edgecolors='w')
    
    plt.title('PCA Projection')
    plt.legend()
    plots['pca'] = get_plot_as_base64()
    plt.close()

    # --- PLOT 4: t-SNE ---
    if len(X_final) > 10:
        try:
            perp = min(30, len(X_final) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
            X_tsne = tsne.fit_transform(X_scaled)

            plt.figure(figsize=(6, 5))
            for cls in z_order:
                indices = np.where(y_final == cls)
                if len(indices[0]) > 0:
                    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                                c=color_map[cls], label=label_map[cls],
                                alpha=0.8, s=40, edgecolors='w')
            
            plt.title('t-SNE Clusters')
            plt.legend()
            plots['tsne'] = get_plot_as_base64()
            plt.close()
        except Exception as e:
            plots['tsne'] = None
            
    return plots