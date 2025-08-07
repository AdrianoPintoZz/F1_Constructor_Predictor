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

def fetch_2025_data():
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
    trainmodels_dir = os.path.abspath(os.path.join(base_dir, '..', 'TrainModels'))
    if not os.path.exists(trainmodels_dir):
        os.makedirs(trainmodels_dir)
    csv_2025_path = os.path.join(trainmodels_dir, 'constructor_2025_dataset.csv')
    df_2025.to_csv(csv_2025_path, index=False)
    print(f"✅ Dados de 2025 salvos em {csv_2025_path} ({df_2025.shape[0]} linhas)")
    return df_2025

# Adiciona ao DataFrame principal apenas se não houver dados de 2025
if __name__ == "__main__":
    # Gera e salva o CSV de 2025 ao rodar o script
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

    # Fill missing final_points with model prediction if not available
    if 'final_points' in final_round_data.columns:
        final_round_data['final_points'] = final_round_data['final_points'].fillna(final_round_data['points'])

    # Prepare features
    final_round_data['team_encoded'] = le_team.transform(final_round_data['team'])
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
        # Get raw probabilities from the model
        probabilities = model.predict_proba(X_year_scaled)
        raw_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]

        # Normalize title_fight_score
        performance_weights = final_round_data['title_fight_score'].values
        performance_weights = performance_weights / (performance_weights.sum() + 1e-8)

        # Combine com pesos mais realistas (85% modelo, 15% performance)
        combined_scores = (raw_probs * 0.85) + (performance_weights * 0.15)

        # Temperatura dinâmica: se disputa equilibrada, temperatura menor (probabilidades mais próximas)
        base_temp = 1.0
        temp_range = 1.5
        temperature = base_temp + temp_range * (1.0 - diff_ratio)
        exp_scores = np.exp(combined_scores * temperature)
        champion_probs = exp_scores / np.sum(exp_scores)

        # Mistura ponderada dos dois primeiros colocados se disputa for muito equilibrada
        if diff_ratio > 0.6:
            idx_top1 = champion_probs.argmax()
            idx_top2 = np.argsort(-champion_probs)[1]
            prob_top1 = champion_probs[idx_top1]
            prob_top2 = champion_probs[idx_top2]
            total = prob_top1 + prob_top2
            champion_probs[idx_top1] = total * 0.65
            champion_probs[idx_top2] = total * 0.35
            # Renormaliza para somar 1
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
    # Feature engineering for 2025 data
    if 'team_encoded' not in df_2025.columns:
        le_team_2025 = LabelEncoder()
        df_2025['team_encoded'] = le_team_2025.fit_transform(df_2025['team'])
    else:
        le_team_2025 = le_team
    # Add missing features with default values if needed
    for col in feature_columns:
        if col not in df_2025.columns:
            df_2025[col] = 0.0
        else:
            df_2025[col] = df_2025[col].fillna(df_2025[col].median())
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


