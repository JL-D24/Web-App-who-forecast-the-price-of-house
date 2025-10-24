import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Web App who forecast the price of house", layout="wide")

# Titre de l'application
st.title("🏠 Web App who forecast the price of house - Comparaison Multi Models")
st.markdown("""
Cette application compare les prédictions de trois modèles différents pour estimer les prix des maisons à Boston(USA).
""")

# Chargement des données
@st.cache_data
def load_data():
    data = pd.read_csv("HousingData.csv")
    data = data.dropna(subset=['MEDV'])
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())
    return data

data = load_data()

# Séparation des features et target
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Préparation du préprocesseur
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Création des modèles
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Régression Linéaire": LinearRegression()
}

# Entraînement des modèles
trained_models = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline

# Interface utilisateur
st.sidebar.header("📋 Paramètres du Bien")

def get_user_input():
    inputs = {}
    
    with st.sidebar.expander("📍 Localisation"):
        inputs['CRIM'] = st.slider('Taux de criminalité (CRIM)', 
                                  float(X['CRIM'].min()), 
                                  float(X['CRIM'].max()), 
                                  float(X['CRIM'].median()))
        inputs['ZN'] = st.slider('Pourcentage zones résidentielles (ZN)', 
                                int(X['ZN'].min()), 
                                int(X['ZN'].max()), 
                                int(X['ZN'].median()))
        inputs['INDUS'] = st.slider('Pourcentage zones industrielles (INDUS)', 
                                   float(X['INDUS'].min()), 
                                   float(X['INDUS'].max()), 
                                   float(X['INDUS'].median()))
        inputs['CHAS'] = st.selectbox('Proximité rivière Charles (CHAS)', [0, 1])
    
    with st.sidebar.expander("🏡 Caractéristiques"):
        inputs['NOX'] = st.slider('Concentration NOx (NOX)', 
                                 float(X['NOX'].min()), 
                                 float(X['NOX'].max()), 
                                 float(X['NOX'].median()))
        inputs['RM'] = st.slider('Nb moy. de pièces (RM)', 
                                float(X['RM'].min()), 
                                float(X['RM'].max()), 
                                float(X['RM'].median()))
        inputs['AGE'] = st.slider('Âge du logement (AGE)', 
                                 float(X['AGE'].min()), 
                                 float(X['AGE'].max()), 
                                 float(X['AGE'].median()))
        inputs['DIS'] = st.slider('Distance centres emploi (DIS)', 
                                 float(X['DIS'].min()), 
                                 float(X['DIS'].max()), 
                                 float(X['DIS'].median()))
    
    with st.sidebar.expander("📊 Autres paramètres"):
        inputs['RAD'] = st.slider('Accessibilité autoroutes (RAD)', 
                                 int(X['RAD'].min()), 
                                 int(X['RAD'].max()), 
                                 int(X['RAD'].median()))
        inputs['TAX'] = st.slider('Taxe foncière (TAX)', 
                                 int(X['TAX'].min()), 
                                 int(X['TAX'].max()), 
                                 int(X['TAX'].median()))
        inputs['PTRATIO'] = st.slider('Ratio élèves/enseignants (PTRATIO)', 
                                     float(X['PTRATIO'].min()), 
                                     float(X['PTRATIO'].max()), 
                                     float(X['PTRATIO'].median()))
        inputs['B'] = st.slider('Proportion population afro-américaine (B)', 
                               float(X['B'].min()), 
                               float(X['B'].max()), 
                               float(X['B'].median()))
        inputs['LSTAT'] = st.slider('% population statut bas (LSTAT)', 
                                   float(X['LSTAT'].min()), 
                                   float(X['LSTAT'].max()), 
                                   float(X['LSTAT'].median()))
    
    df = pd.DataFrame(columns=X.columns)
    for col in X.columns:
        if col in inputs:
            df[col] = [inputs[col]]
        else:
            if X[col].dtype == 'object':
                df[col] = [X[col].mode()[0]]
            else:
                df[col] = [X[col].median()]
    return df

user_input = get_user_input()

# Affichage des inputs
st.subheader("📌 Paramètres sélectionnés")
st.write(user_input)

# Bouton de prédiction
if st.sidebar.button("🔮 Comparer les prédictions"):
    st.subheader("📊 Résultats des prédictions")
    
    # Prédictions et métriques
    results = []
    predictions = {}
    
    for name, model in trained_models.items():
        # Prédiction sur l'input utilisateur
        pred = model.predict(user_input)[0]
        predictions[name] = pred
        
        # Prédiction sur le test set pour les métriques
        y_pred = model.predict(X_test)
        
        results.append({
            "Modèle": name,
            "Prédiction ($1000)": f"{pred:.2f}",
            "R²": f"{r2_score(y_test, y_pred):.4f}",
            "Erreur Moyenne Absolue": f"{mean_absolute_error(y_test, y_pred):.2f}"
        })
    
    # Affichage des résultats sous forme de tableau
    results_df = pd.DataFrame(results)
    st.table(results_df)
    
    # Graphique comparatif des prédictions
    st.subheader("📈 Comparaison des prédictions")
    fig, ax = plt.subplots()
    models = list(predictions.keys())
    pred_values = list(predictions.values())
    ax.bar(models, pred_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Prix Prédit ($1000)')
    ax.set_title('Comparaison des prédictions entre modèles')
    st.pyplot(fig)
    
    # Graphique de performance des modèles
    st.subheader("🏆 Performance des modèles")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots()
        r2_scores = [float(x['R²']) for x in results]
        ax1.bar(models, r2_scores, color='#1f77b4')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score R²')
        ax1.set_title('Comparaison des scores R²')
        st.pyplot(fig1)
    
    with col2:
        fig2, ax2 = plt.subplots()
        mae_scores = [float(x['Erreur Moyenne Absolue']) for x in results]
        ax2.bar(models, mae_scores, color='#ff7f0e')
        ax2.set_ylabel('Erreur Moyenne Absolue')
        ax2.set_title('Comparaison des erreurs MAE')
        st.pyplot(fig2)
    
    # Importance des features pour Random Forest
    st.subheader("🔍 Importance des caractéristiques (Random Forest)")
    try:
        rf_model = trained_models["Random Forest"].named_steps['regressor']
        feature_importances = rf_model.feature_importances_
        
        # Get feature names
        preprocessor = trained_models["Random Forest"].named_steps['preprocessor']
        preprocessor.fit(X_train)
        
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        num_feature_names = list(numeric_features)
        
        if len(categorical_features) > 0:
            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
        else:
            cat_feature_names = []
        
        all_feature_names = num_feature_names + cat_feature_names
        
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False).head(10)
        
        st.bar_chart(importance_df.set_index('Feature'))
        st.write("Détails de l'importance des caractéristiques :")
        st.dataframe(importance_df)
        
    except Exception as e:
        st.warning(f"Impossible d'afficher l'importance des features: {str(e)}")

# Section d'exploration des données
if st.checkbox("🔍 Afficher l'exploration des données"):
    st.subheader("📂 Exploration du Dataset")
    
    tab1, tab2, tab3 = st.tabs(["Données Brutes", "Statistiques", "Visualisation"])
    
    with tab1:
        st.write(data.head())
    
    with tab2:
        st.write(data.describe())
    
    with tab3:
        selected_col = st.selectbox("Sélectionnez une colonne à visualiser", data.columns)
        
        if data[selected_col].dtype == 'object':
            st.bar_chart(data[selected_col].value_counts())
        else:
            st.line_chart(data[selected_col])
