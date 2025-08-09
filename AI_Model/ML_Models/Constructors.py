import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
import sys

from fastf1 import get_session
from fastf1.events import get_event_schedule
warnings.filterwarnings('ignore')

print("🏎️  F1 Constructor Championship Prediction Model")
print("=" * 55)

# Load the optimized dataset
base_dir = os.path.dirname(__file__)
csv_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'TrainModels', 'constructor_ml_dataset.csv'))
validation_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'TrainModels', 'championship_standings.csv'))


df = pd.read_csv(csv_path)


standings_df = pd.read_csv(validation_path)


le_team = LabelEncoder()
df['team_encoded'] = le_team.fit_transform(df['team'])



# 1. Consistency metrics (HIGH PRIORITY - champions are consistent)
df['grid_consistency'] = df.groupby(['year', 'team'])['avg_grid_position'].transform('std').fillna(0)
df['finish_consistency'] = df.groupby(['year', 'team'])['avg_finish_position'].transform('std').fillna(0)
df['points_consistency'] = df.groupby(['year', 'team'])['points_per_race'].transform('std').fillna(0)

# Consistency score (lower std = more consistent = better)
df['consistency_score'] = 1 / (1 + df['grid_consistency'] + df['finish_consistency'] + df['points_consistency'])

# 2. Performance trend (HIGH PRIORITY - improving performance)
df['points_trend'] = df.groupby(['year', 'team'])['points'].diff().fillna(0)
df['position_improvement'] = df.groupby(['year', 'team'])['avg_finish_position'].diff().fillna(0) * -1  # Negative because lower position is better

# 3. Elite performance indicators (HIGHEST PRIORITY)
df['top3_rate'] = np.where(df['avg_finish_position'] <= 3, 1, 0)
df['podium_consistency'] = df.groupby(['year', 'team'])['top3_rate'].transform('mean')

# 4. Competitive advantage metrics (HIGH PRIORITY)
df['points_dominance'] = df['points_per_race'] / (df.groupby(['year', 'round'])['points_per_race'].transform('mean') + 1)
df['grid_advantage'] = 20 - df['avg_grid_position']  # Higher = better grid position
df['race_advantage'] = 20 - df['avg_finish_position']  # Higher = better race position

# 5. Advanced performance ratios
df['grid_to_finish_ratio'] = df['avg_grid_position'] / (df['avg_finish_position'] + 1)  # >1 means losing positions
df['points_efficiency'] = df['points_per_race'] / (21 - df['avg_finish_position'])  # Points per expected position

# 6. NEW: Season progression metrics (CRITICAL FOR LATE SEASON PREDICTIONS)
df['season_progress'] = df['round'] / 24.0  # Normalize round to 0-1
df['late_season_performance'] = np.where(df['round'] >= 15, df['points_per_race'] * 1.5, df['points_per_race'])
df['championship_momentum'] = df.groupby(['year', 'team'])['points_per_race'].transform(lambda x: x.rolling(3, min_periods=1).mean())

# 7. Position-based scoring that heavily favors actual championship contenders
df['championship_contender'] = np.where(df['points_per_race'] > df.groupby(['year', 'round'])['points_per_race'].transform('quantile', 0.7), 1, 0)
df['title_fight_score'] = df['points_per_race'] * df['consistency_score'] * (1 + df['season_progress'])

# Select features with CORRECTED priority weighting
feature_columns = [
    'round', 'team_encoded',
    # CRITICAL - These should be the strongest predictors
    'title_fight_score',        # Weight: 4.0 - NEW: Best overall championship indicator
    'late_season_performance',  # Weight: 3.5 - NEW: Critical for final predictions
    'championship_momentum',    # Weight: 3.0 - NEW: Recent form matters most
    # HIGHEST PRIORITY - Direct performance
    'points_per_race',          # Weight: 3.0
    'championship_contender',   # Weight: 2.8 - NEW: Are they actually in the fight?
    'final_points',             # Weight: 3.5 - NEW: True final points for realism
    # HIGH PRIORITY - Consistency and position
    'podium_consistency',       # Weight: 2.5
    'consistency_score',        # Weight: 2.2
    'avg_finish_position',      # Weight: 2.0 - More important than grid
    # MEDIUM PRIORITY - Supporting metrics
    'race_advantage',           # Weight: 1.8
    'points_dominance',         # Weight: 1.5
    'reliability_rate',         # Weight: 1.5
    'season_progress',          # Weight: 1.2 - NEW: Time in season matters
    # LOWER PRIORITY - Less critical
    'avg_grid_position',        # Weight: 1.0 - Reduced importance
    'recovery_rate',            # Weight: 1.0
    'grid_advantage'            # Weight: 0.8 - Least important
]

# Create CORRECTED feature weights dictionary
feature_weights = {
    'round': 1.0,
    'team_encoded': 1.0,
    'title_fight_score': 3.5,          
    'late_season_performance': 2.5,    
    'championship_momentum': 3.0,      
    'points_per_race': 2.8,            
    'championship_contender': 2.8,     
    'final_points': 3.5,               
    'podium_consistency': 3.0,         
    'consistency_score': 1.8,          
    'avg_finish_position': 3.5,        
    'race_advantage': 2.0,
    'points_dominance': 2.0,
    'reliability_rate': 2.0,           
    'season_progress': 1.5,
    'avg_grid_position': 2.0,
    'recovery_rate': 2.0,              
    'grid_advantage': 1.5
}


# Handle any missing values in new features
for col in feature_columns:
    if col not in ['round', 'team_encoded']:
        df[col] = df[col].fillna(df[col].median())


X = df[feature_columns]
y = df['is_champion']


print(f"🎯 Target variable: is_champion")



train_data = df[df['year'].isin([2021, 2022, 2023])].copy()
test_data = df[df['year'] == 2024].copy()

X_train = train_data[feature_columns]
y_train = train_data['is_champion']
X_test = test_data[feature_columns] 
y_test = test_data['is_champion']

print(f"🚂 Training data: {X_train.shape[0]} samples from 2021-2023")
print(f"🧪 Test data: {X_test.shape[0]} samples from 2024")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


sample_weights = np.where(y_train == 1, 3.0, 1.0)  

rf_model = RandomForestClassifier(
    n_estimators=700,           # More trees for better performance
    max_depth=15,              # Deeper trees to capture complex patterns
    min_samples_split=3,       # More sensitive to patterns
    min_samples_leaf=1,        # Allow finer decision boundaries
    max_features='sqrt',       # Good balance for feature selection
    random_state=42,
    class_weight='balanced',   # Handle imbalanced classes
    bootstrap=True,
    oob_score=True            # Out-of-bag scoring for validation
)

rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)



cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')


# Feature importance with priority analysis
print(f"\n🔍 Feature importance (with priority weights):")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_,
    'priority_weight': [feature_weights.get(f, 1.0) for f in feature_columns]
}).sort_values('importance', ascending=False)

feature_importance['weighted_importance'] = feature_importance['importance'] * feature_importance['priority_weight']
feature_importance = feature_importance.sort_values('weighted_importance', ascending=False)

print(f"{'Feature':<25} {'Raw Imp':<8} {'Priority':<8} {'Weighted':<10} {'Impact'}")
print("-" * 70)
for _, row in feature_importance.head(10).iterrows():
    priority_icon = "🔴" if row['priority_weight'] >= 2.5 else "🟠" if row['priority_weight'] >= 2.0 else "🟡" if row['priority_weight'] >= 1.5 else "🟢"
    print(f"{row['feature']:<25} {row['importance']:<8.3f} {row['priority_weight']:<8.1f} {row['weighted_importance']:<10.3f} {priority_icon}")

# Show top performing features
top_features = feature_importance.head(5)['feature'].tolist()
print(f"\n🏆 Top 5 most impactful features: {', '.join(top_features)}")

# Predictions on test set
print(f"\n🎯 Making predictions on 2024 data...")
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)

# Test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Test accuracy: {test_accuracy:.3f}")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Helper: compute engineered features on a dataset that has the base columns
def compute_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_in.copy()
    # Ensure required base columns exist
    base_needed = [
        'year', 'round', 'team', 'final_points', 'points',
        'avg_grid_position', 'avg_finish_position', 'points_per_race',
        'reliability_rate'
    ]
    for col in base_needed:
        if col not in df_feat.columns:
            # Provide neutral defaults to avoid NaNs
            if col in ['year', 'round']:
                df_feat[col] = 0
            elif col == 'team':
                df_feat[col] = 'Unknown'
            else:
                df_feat[col] = 0.0

    # 1. Consistency metrics
    df_feat['grid_consistency'] = df_feat.groupby(['year', 'team'])['avg_grid_position'].transform('std').fillna(0)
    df_feat['finish_consistency'] = df_feat.groupby(['year', 'team'])['avg_finish_position'].transform('std').fillna(0)
    df_feat['points_consistency'] = df_feat.groupby(['year', 'team'])['points_per_race'].transform('std').fillna(0)
    df_feat['consistency_score'] = 1 / (1 + df_feat['grid_consistency'] + df_feat['finish_consistency'] + df_feat['points_consistency'])

    # 2. Performance trend
    df_feat['points_trend'] = df_feat.groupby(['year', 'team'])['points'].diff().fillna(0)
    df_feat['position_improvement'] = df_feat.groupby(['year', 'team'])['avg_finish_position'].diff().fillna(0) * -1

    # 3. Elite performance indicators
    df_feat['top3_rate'] = np.where(df_feat['avg_finish_position'] <= 3, 1, 0)
    df_feat['podium_consistency'] = df_feat.groupby(['year', 'team'])['top3_rate'].transform('mean').fillna(0)

    # 4. Competitive advantage metrics
    df_feat['points_dominance'] = df_feat['points_per_race'] / (df_feat.groupby(['year', 'round'])['points_per_race'].transform('mean') + 1)
    df_feat['grid_advantage'] = 20 - df_feat['avg_grid_position']
    df_feat['race_advantage'] = 20 - df_feat['avg_finish_position']

    # 5. Advanced performance ratios
    df_feat['grid_to_finish_ratio'] = df_feat['avg_grid_position'] / (df_feat['avg_finish_position'] + 1)
    df_feat['points_efficiency'] = df_feat['points_per_race'] / (21 - df_feat['avg_finish_position'])

    # 6. Season progression
    if 'total_rounds' in df_feat.columns and (df_feat['total_rounds'] > 0).any():
        df_feat['season_progress'] = df_feat['round'] / df_feat['total_rounds'].replace(0, np.nan)
        df_feat['season_progress'] = df_feat['season_progress'].fillna(0)
    else:
        df_feat['season_progress'] = df_feat['round'] / 24.0
    df_feat['late_season_performance'] = np.where(df_feat['round'] >= 15, df_feat['points_per_race'] * 1.5, df_feat['points_per_race'])
    df_feat['championship_momentum'] = df_feat.groupby(['year', 'team'])['points_per_race'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # 7. Contender flag and title score
    rr_mean = df_feat.groupby(['year', 'round'])['points_per_race'].transform('quantile', 0.7)
    df_feat['championship_contender'] = np.where(df_feat['points_per_race'] > rr_mean, 1, 0)
    df_feat['title_fight_score'] = df_feat['points_per_race'] * df_feat['consistency_score'] * (1 + df_feat['season_progress'])

    # Final NA cleanup for model features
    for col in feature_columns:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
        else:
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    return df_feat

def fetch_2025_data(force_refresh: bool = False):
    """Fetch 2025 data from FastF1 API and cache to CSV.

    If the CSV already exists and force_refresh is False, reuse it and skip the API.
    """
    # CSV cache path (kept consistent with the path used later for reading)
    trainmodels_dir = os.path.abspath(os.path.join(base_dir, '..', 'TrainModels'))
    if not os.path.exists(trainmodels_dir):
        os.makedirs(trainmodels_dir)
    csv_2025_path = os.path.join(trainmodels_dir, 'constructor_2025_dataset.csv')

    # If cache exists and we aren't forcing a refresh, reuse it
    if (not force_refresh) and os.path.exists(csv_2025_path):
        try:
            cached_df = pd.read_csv(csv_2025_path)
            print(f"📦 Usando CSV 2025 já existente: {csv_2025_path} ({cached_df.shape[0]} linhas)")
            return cached_df
        except Exception as e:
            print(f"⚠️ Falha ao ler CSV existente ({csv_2025_path}). Tentando buscar via API... Detalhes: {e}")

    schedule = get_event_schedule(2025)
    if schedule.empty:
        print("⚠️ Nenhum evento disponível para 2025 na API FastF1.")
        return None
    data_2025 = []
    total_rounds = schedule['RoundNumber'].max()
    team_stats = {}
    for _, event in schedule.iterrows():
        round_num = event['RoundNumber']
        try:
            session = get_session(2025, round_num, 'R')
            session.load()
            results = session.results
            teams = results['TeamName'].unique()
            for team in teams:
                if team not in team_stats:
                    team_stats[team] = {
                        'points': 0,
                        'grid_positions': [],
                        'finish_positions': [],
                        'positions_gained': 0,
                        'dnfs': 0,
                        'races': 0
                    }
                team_results = results[results['TeamName'] == team]
                team_stats[team]['points'] += team_results['Points'].sum()
                grid_positions = team_results['GridPosition'].fillna(20)
                finish_positions = team_results['Position'].fillna(20)
                team_stats[team]['grid_positions'].extend(grid_positions.tolist())
                team_stats[team]['finish_positions'].extend(finish_positions.tolist())
                team_stats[team]['positions_gained'] += (grid_positions - finish_positions).sum()
                team_stats[team]['dnfs'] += (team_results['Status'] != 'Finished').sum()
                team_stats[team]['races'] += 1
            for team in teams:
                stats = team_stats[team]
                races = stats['races'] if stats['races'] > 0 else 1
                avg_grid_pos = np.mean(stats['grid_positions']) if stats['grid_positions'] else 20
                avg_finish_pos = np.mean(stats['finish_positions']) if stats['finish_positions'] else 20
                entry = {
                    'year': 2025,
                    'round': round_num,
                    'total_rounds': total_rounds,
                    'season_progress': round_num / total_rounds,
                    'team': team,
                    'champion': None,
                    'is_champion': 0,
                    'final_points': stats['points'],
                    'points': stats['points'],
                    'avg_grid_position': avg_grid_pos,
                    'avg_finish_position': avg_finish_pos,
                    'positions_gained': stats['positions_gained'],
                    'fastest_laps': 0,
                    'best_lap_time': 0,
                    'dnfs': stats['dnfs'],
                    'recovery_rate': avg_grid_pos - avg_finish_pos,
                    'points_per_race': stats['points'] / races,
                    'reliability_rate': 1 - (stats['dnfs'] / (races * 2))
                }
                data_2025.append(entry)
        except Exception as e:
            print(f"⚠️ Erro ao buscar dados do round {round_num} de 2025: {e}")
            continue
    df_2025 = pd.DataFrame(data_2025)
    # Salva o CSV de 2025
    df_2025.to_csv(csv_2025_path, index=False)
    print(f"✅ Dados de 2025 salvos em {csv_2025_path} ({df_2025.shape[0]} linhas)")
    return df_2025

# Adiciona ao DataFrame principal apenas se não houver dados de 2025
if __name__ == "__main__":
    # Gera e salva o CSV de 2025 apenas se não existir (usa cache caso exista)
    tm_dir = os.path.abspath(os.path.join(base_dir, '..', 'TrainModels'))
    csv_2025_path_main = os.path.join(tm_dir, 'constructor_2025_dataset.csv')
    if os.path.exists(csv_2025_path_main):
        try:
            df_cached = pd.read_csv(csv_2025_path_main)
            print(f"📦 CSV 2025 já existe: {csv_2025_path_main} ({df_cached.shape[0]} linhas). Pulando fetch da API.")
        except Exception as e:
            print(f"⚠️ CSV 2025 existe mas não pôde ser lido. Tentando refazer via API... Detalhes: {e}")
            fetch_2025_data(force_refresh=True)
    else:
        fetch_2025_data()


# Function to predict championship probabilities for each year
def predict_championship_probabilities(year_data, model, scaler, le_team, feature_cols):
    """Predict championship probabilities for all teams in a given year"""
    
    # Get the final round data for each team (latest round per team)
    final_round_data = year_data.loc[year_data.groupby('team')['round'].idxmax()].copy()

    # Merge com standings_df para garantir pontos finais reais
    global standings_df
    final_round_data = final_round_data.merge(
        standings_df[['year', 'team', 'final_points']],
        on=['year', 'team'],
        how='left',
        suffixes=('', '_real')
    )

    # Preferir pontos finais reais do standings, depois existentes, depois pontos acumulados
    if 'final_points_real' in final_round_data.columns:
        # Usa final_points_real quando disponível; senão mantém o existente; senão usa 'points'
        existing_fp = final_round_data['final_points'] if 'final_points' in final_round_data.columns else np.nan
        final_round_data['final_points'] = (
            final_round_data['final_points_real']
            .combine_first(existing_fp)
            .fillna(final_round_data['points'])
        )
    else:
        if 'final_points' in final_round_data.columns:
            final_round_data['final_points'] = final_round_data['final_points'].fillna(final_round_data['points'])

    # Prepare features
    # Safe team encoding: map known classes; unseen teams go to a reserved code (max+1)
    try:
        known_classes = list(getattr(le_team, 'classes_', []))
    except Exception:
        known_classes = []
    mapping = {cls: idx for idx, cls in enumerate(known_classes)}
    reserved_code = (max(mapping.values()) + 1) if mapping else 0
    final_round_data['team_encoded'] = final_round_data['team'].map(mapping).fillna(reserved_code).astype(int)
    X_year = final_round_data[feature_cols]
    X_year_scaled = scaler.transform(X_year)

    # Calcular diferença de pontos entre campeão e segundo colocado
    sorted_points = final_round_data.sort_values('final_points', ascending=False)
    if len(sorted_points) > 1:
        diff_points = sorted_points.iloc[0]['final_points'] - sorted_points.iloc[1]['final_points']
    else:
        diff_points = 0.0
    max_points = sorted_points.iloc[0]['final_points'] if len(sorted_points) > 0 else 1.0
    diff_ratio = 1.0 - min(diff_points / (max_points + 1e-8), 1.0)

    try:
        # Probabilidades do modelo
        probabilities = model.predict_proba(X_year_scaled)
        ml_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

        # Converte para logits e normaliza (evita saturação)
        ml_probs = np.clip(ml_probs, 1e-6, 1 - 1e-6)
        ml_logits = np.log(ml_probs / (1 - ml_probs))
        def _zscore(x: np.ndarray) -> np.ndarray:
            x = x.astype(float)
            return (x - x.mean()) / (x.std() + 1e-6)

        ml_component = _zscore(ml_logits)

        # Prior baseado em performance (pontos finais preferencialmente; se não, points_per_race)
        eps = 1e-8
        fp_arr = final_round_data['final_points'].to_numpy(dtype=float)
        ppr_arr = final_round_data['points_per_race'].to_numpy(dtype=float) if 'points_per_race' in final_round_data.columns else np.zeros_like(fp_arr)
        if fp_arr.sum() > 0:
            prior_base = fp_arr
        elif ppr_arr.sum() > 0:
            prior_base = ppr_arr
        else:
            prior_base = np.ones_like(fp_arr)
        prior_norm = prior_base / (prior_base.sum() + eps)

        # Peso dinâmico do prior cresce ao longo da temporada; também acentua separação com potência
        if 'season_progress' in final_round_data.columns:
            season_prog_mean = float(final_round_data['season_progress'].mean())
        else:
            # fallback: usa round/24
            season_prog_mean = float((final_round_data['round'].max() or 0) / 24.0)
        season_prog_mean = max(0.0, min(1.0, season_prog_mean))

        # Detecta se os pontos finais são "reais" (vindos do standings) ou apenas acumulados até o momento
        has_real_final_points = 'final_points_real' in final_round_data.columns and final_round_data['final_points_real'].notna().any()

        # Ajusta progressão efetiva: se não temos pontos finais reais (dados em curso), reduza convicção
        effective_prog = season_prog_mean if has_real_final_points else season_prog_mean * 0.75

        # Potência do prior: mais baixa quando temporada não finalizada
        # ~1.0 (início) a ~2.2 (fim) quando não-final; ~1.2 a ~2.6 quando final
        if has_real_final_points:
            gamma = 1.2 + 1.4 * effective_prog
        else:
            gamma = 1.0 + 1.2 * effective_prog
        prior_scores = np.power(prior_norm + eps, gamma)
        prior_scores = prior_scores / (prior_scores.sum() + eps)
        prior_component = _zscore(np.log(prior_scores + eps))

        # Combinação: no início, dá mais peso ao modelo; no fim, mais ao prior de pontos
        if has_real_final_points:
            w_prior = 0.25 + 0.55 * effective_prog   # 0.25 -> 0.80
        else:
            w_prior = 0.20 + 0.40 * effective_prog   # 0.20 -> 0.60 (mais conservador)
        w_prior = float(max(0.0, min(0.95, w_prior)))
        w_ml = 1.0 - w_prior
        combined = (w_ml * ml_component) + (w_prior * prior_component)

        # Sharpening com base na vantagem em pontos (maior vantagem => mais nítido)
        gap_ratio = float(diff_points / (max_points + eps)) if (max_points + eps) > 0 else 0.0
        if has_real_final_points:
            # Final de temporada: evitar picos exagerados mesmo com grande gap
            tau = 1.30 + 2.00 * gap_ratio  # até ~3.30 antes do cap
        else:
            # Temporada em curso: ainda mais suave
            tau = 1.30 + 2.00 * gap_ratio  # até ~3.30 antes do cap
        tau = float(max(1.02, min(2.8 if has_real_final_points else 3.2, tau)))
        exp_scores = np.exp(combined * tau)
        champion_probs = exp_scores / (exp_scores.sum() + eps)

        # Mistura com uma distribuição uniforme quando a temporada ainda não está madura
        # Reduz confiança excessiva em dados em curso (ex.: 2025 parcial)
        if not has_real_final_points:
            k = len(champion_probs)
            # Use distribuição baseada em pontos (prior_scores) para preservar separação
            prior_dist = prior_scores.astype(float)
            prior_dist = prior_dist / (prior_dist.sum() + eps)
            # incerteza decai linearmente até 0 por volta de 60% da temporada
            uncertainty = float(max(0.0, 0.6 - effective_prog) / 0.6)
            blend_alpha = 0.50 * uncertainty  # até 50% de mistura no começo
            champion_probs = (1.0 - blend_alpha) * champion_probs + blend_alpha * prior_dist
            champion_probs = champion_probs / (champion_probs.sum() + eps)

            # Garantir que o topo não ultrapasse um teto dinâmico em temporada em curso
            top = float(champion_probs.max())
            max_top_allowed = 0.60 + 0.20 * effective_prog  # 60% no início -> 80% mais tarde
            if top > max_top_allowed:
                alpha_extra = float(min(0.7, (top - max_top_allowed) * 1.8))
                champion_probs = (1.0 - alpha_extra) * champion_probs + alpha_extra * prior_dist
                champion_probs = champion_probs / (champion_probs.sum() + eps)
        else:
            # Temporada finalizada: pequena suavização para evitar 98%+
            k = len(champion_probs)
            if k > 1:
                uniform = np.ones(k, dtype=float) / k
                blend_alpha_final = 0.08
                champion_probs = (1.0 - blend_alpha_final) * champion_probs + blend_alpha_final * uniform
                champion_probs = champion_probs / (champion_probs.sum() + eps)

            # Limite suave para prob máxima em temporada finalizada baseado no gap
            top = float(champion_probs.max())
            max_top_allowed_final = 0.70 + 0.25 * float(min(1.0, max(0.0, gap_ratio)))  # 70% -> 95% (teórico)
            max_top_allowed_final = float(min(0.90, max_top_allowed_final))  # teto absoluto 90%
            if top > max_top_allowed_final and k > 1:
                alpha_extra_final = float(min(0.5, (top - max_top_allowed_final) * 1.5))
                champion_probs = (1.0 - alpha_extra_final) * champion_probs + alpha_extra_final * uniform
        champion_probs = champion_probs / (champion_probs.sum() + eps)

        # Piso mínimo para evitar quase-uniforme em times sem chance e renormalização
        floor = 0.001
        champion_probs = np.maximum(champion_probs, floor)
        champion_probs = champion_probs / champion_probs.sum()

    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        champion_probs = np.ones(len(final_round_data)) / len(final_round_data)

    results = pd.DataFrame({
        'team': final_round_data['team'],
        'final_points': final_round_data['final_points'],
        'championship_probability': champion_probs,
        'predicted_champion': champion_probs == champion_probs.max(),
        'title_fight_score': final_round_data['title_fight_score'],
        'diff_points_to_2nd': diff_points,
        'diff_ratio': diff_ratio
    }).sort_values('championship_probability', ascending=False)

    return results


# Make predictions for all years
print(f"\n🔮 Championship Predictions by Year")
print("=" * 50)

all_predictions = []

for year in sorted(df['year'].unique()):
    print(f"\n📅 Year {year}:")
    
    # Get year data
    year_data = df[df['year'] == year].copy()
    
    # Get actual champion from standings
    actual_champion = standings_df[standings_df['year'] == year].loc[
        standings_df[standings_df['year'] == year]['position'] == 1, 'team'
    ].iloc[0]
    
    # Predict probabilities
    predictions = predict_championship_probabilities(
        year_data, rf_model, scaler, le_team, feature_columns
    )
    
    # Get predicted champion
    predicted_champion = predictions.loc[predictions['predicted_champion'], 'team'].iloc[0]
    
    print(f"🏆 Actual Champion: {actual_champion}")
    print(f"🤖 Predicted Champion: {predicted_champion}")
    print(f"✅ Prediction Correct: {'Yes' if actual_champion == predicted_champion else 'No'}")
    
    print(f"\n🎯 Championship Probabilities:")
    for _, row in predictions.head(5).iterrows():  # Show top 5
        prob_percent = row['championship_probability'] * 100
        star = "⭐" if row['predicted_champion'] else "  "
        title_score = row['title_fight_score']
        print(f"{star} {row['team']:<20} {prob_percent:>6.2f}% ({row['final_points']:>3.0f} pts, score: {title_score:.2f})")
    
    # Verify probability sum (should be exactly 100%)
    all_probs_sum = predictions['championship_probability'].sum() * 100
    print(f"   {'Total probabilities:':<20} {all_probs_sum:>6.2f}%")
    
    # Safety check - warn if probabilities don't sum to 100%
    if abs(all_probs_sum - 100.0) > 0.01:
        print(f"⚠️  WARNING: Probabilities don't sum to 100% (sum = {all_probs_sum:.2f}%)")
    
    # Store for summary
    all_predictions.append({
        'year': year,
        'actual_champion': actual_champion,
        'predicted_champion': predicted_champion,
        'correct': actual_champion == predicted_champion,
        'top_probability': predictions['championship_probability'].max()
    })

# Summary statistics
print(f"\n📈 Model Performance Summary")
print("=" * 40)

predictions_df = pd.DataFrame(all_predictions)
overall_accuracy = predictions_df['correct'].mean()
print(f"🎯 Overall Championship Prediction Accuracy: {overall_accuracy:.1%}")

print(f"\n📊 Year-by-Year Results:")
for _, row in predictions_df.iterrows():
    status = "✅" if row['correct'] else "❌"
    prob = row['top_probability'] * 100
    print(f"{row['year']}: {status} {row['predicted_champion']} (confidence: {prob:.1f}%)")

# Confidence analysis
print(f"\n🔍 Prediction Confidence Analysis:")
high_confidence = predictions_df[predictions_df['top_probability'] > 0.7]
print(f"High confidence predictions (>70%): {len(high_confidence)}/{len(predictions_df)}")
if len(high_confidence) > 0:
    high_conf_accuracy = high_confidence['correct'].mean()
    print(f"Accuracy on high confidence predictions: {high_conf_accuracy:.1%}")

print(f"\n💾 Model training completed successfully!")
print(f"🏁 Ready to make predictions for future seasons!")

# Prever 2025 com os dados do novo CSV
csv_2025_path = os.path.abspath(os.path.join(base_dir, '..', 'TrainModels', 'constructor_2025_dataset.csv'))
if os.path.exists(csv_2025_path):
    print(f"\n🔮 Previsão para 2025 usando o novo CSV:")
    df_2025 = pd.read_csv(csv_2025_path)
    # Compute engineered features for 2025 data to avoid zeros and uniform outputs
    df_2025 = compute_engineered_features(df_2025)
    # Use training label encoder; unseen teams handled inside predictor via safe mapping
    le_team_2025 = le_team
    # Predict using last round for each team
    predictions_2025 = predict_championship_probabilities(
        df_2025, rf_model, scaler, le_team_2025, feature_columns
    )
    print(f"\n🎯 Probabilidades de campeonato para 2025:")
    for _, row in predictions_2025.iterrows():
        prob_percent = row['championship_probability'] * 100
        star = "⭐" if row['predicted_champion'] else "  "
        title_score = row['title_fight_score']
        print(f"{star} {row['team']:<20} {prob_percent:>6.2f}% ({row['final_points']:>3.0f} pts, score: {title_score:.2f})")
    all_probs_sum = predictions_2025['championship_probability'].sum() * 100
    print(f"   {'Total probabilities:':<20} {all_probs_sum:>6.2f}%")
    predicted_champion_2025 = predictions_2025.loc[predictions_2025['predicted_champion'], 'team'].iloc[0]
    print(f"\n🤖 Campeão previsto para 2025: {predicted_champion_2025}")
else:
    print(f"\n⚠️  O arquivo de dados para 2025 não foi encontrado em {csv_2025_path}!")


