from fastf1 import get_session, Cache
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# Set up cache directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
cache_dir = os.path.join(base_dir, 'cache')
Cache.enable_cache(cache_dir)

# Configuration
YEARS = [2021, 2022, 2023, 2024]
TEST_ROUNDS = [2, 4, 6, 8, 10, 13, 15, 18, 20, 22]  # Expandido para incluir mais corridas de 2024

# Total rounds per season (important for model context)
TOTAL_ROUNDS = {
    2021: 22,  # 22 rounds in 2021
    2022: 22,  # 22 rounds in 2022  
    2023: 23,  # 23 rounds in 2023
    2024: 24   # 24 rounds in 2024
}

# Cache to store race data and avoid reloading
race_cache = {}

def get_team_stats(session, team):
    stats = defaultdict(float)
    team_drivers = session.results[session.results['TeamName'] == team]
    
    if not team_drivers.empty:
        stats['points'] = team_drivers['Points'].sum()
        
        # Position statistics
        grid_positions = team_drivers['GridPosition'].fillna(20)  # Handle missing grid positions
        finish_positions = team_drivers['Position'].fillna(20)   # Handle DNFs
        
        stats['avg_grid_pos'] = grid_positions.mean()
        stats['avg_finish_pos'] = finish_positions.mean()
        stats['positions_gained'] = (grid_positions - finish_positions).sum()
        
        # Performance metrics
        stats['fastest_laps'] = team_drivers['Points'].apply(lambda x: 1 if x > 0 and x % 1 != 0 else 0).sum()
        
        # Reliability metrics
        stats['dnfs'] = (team_drivers['Status'] != 'Finished').sum()
        
        # Try to get best lap time
        try:
            driver_numbers = team_drivers['DriverNumber'].tolist()
            if driver_numbers:
                team_laps = session.laps[session.laps['DriverNumber'].isin(driver_numbers)]
                if not team_laps.empty:
                    valid_laps = team_laps.dropna(subset=['LapTime'])
                    if not valid_laps.empty:
                        stats['best_lap_time'] = valid_laps['LapTime'].min().total_seconds()
        except Exception:
            stats['best_lap_time'] = np.nan
            
    return stats

def get_final_standings(year):
    # Dados estáticos para cada temporada
    static_standings = {
        2021: {
            'Red Bull Racing': 585,
            'Mercedes': 613,
            'Ferrari': 323,
            'McLaren': 275,
            'Alpine': 155,
            'AlphaTauri': 142,
            'Aston Martin': 77,
            'Williams': 23,
            'Alfa Romeo': 13,
            'Haas F1 Team': 0
        },
        2022: {
            'Red Bull Racing': 759,
            'Ferrari': 554,
            'Mercedes': 515,
            'Alpine': 173,
            'McLaren': 159,
            'Alfa Romeo': 55,
            'Aston Martin': 55,
            'Haas F1 Team': 37,
            'AlphaTauri': 35,
            'Williams': 8
        },
        2023: {
            'Red Bull Racing': 860,
            'Mercedes': 409,
            'Ferrari': 406,
            'McLaren': 302,
            'Aston Martin': 280,
            'Alpine': 120,
            'Williams': 28,
            'AlphaTauri': 25,
            'Alfa Romeo': 16,
            'Haas F1 Team': 12
        },
        2024: {
            'McLaren': 666,
            'Ferrari': 652,
            'Red Bull Racing': 589,
            'Mercedes': 468,
            'Aston Martin': 94,
            'Alpine': 65,
            'Haas F1 Team': 58,
            'RB': 46,
            'Williams': 17,
            'Alfa Romeo': 4,
        }
    }
    print(f"🔍 Getting static final constructor standings for {year}...")
    standings = static_standings.get(year, {})
    if standings:
        champion_team = max(standings.items(), key=lambda x: x[1])[0]
        print(f"🏆 Final constructor championship for {year}: {champion_team}")
        for i, (team, points) in enumerate(sorted(standings.items(), key=lambda x: x[1], reverse=True), 1):
            champion_mark = " 🏆" if team == champion_team else ""
            print(f"  {i}. {team}: {points} points{champion_mark}")
        return champion_team, standings
    else:
        print(f"❌ No static constructor standings found for {year}")
        return None, {}

def get_accumulated_stats(year, until_round, needed_rounds):
    """Collect accumulated statistics for all teams up to a specific round - ONLY using TEST_ROUNDS"""
    team_stats = defaultdict(lambda: defaultdict(float))
    race_count = defaultdict(int)
    
    # Only load the rounds we actually need from TEST_ROUNDS that are <= until_round
    rounds_to_load = [r for r in needed_rounds if r <= until_round]
    
    for round_num in rounds_to_load:
        cache_key = f"{year}_{round_num}"
        
        try:
            # Check if we already have this race data cached
            if cache_key not in race_cache:
                session = get_session(year, round_num, 'R')
                session.load()
                race_cache[cache_key] = session
                print(f"  Loaded race {round_num} for {year} (needed: {rounds_to_load})")
            else:
                session = race_cache[cache_key]
            
            # Collect unique teams
            teams = session.results['TeamName'].unique()
            
            # Collect statistics for each team
            for team in teams:
                normalized_team = normalize_team_name(team.strip())
                stats = get_team_stats(session, team)
                race_count[normalized_team] += 1
                
                # Accumulate all statistics
                for key, value in stats.items():
                    if not pd.isna(value):
                        team_stats[normalized_team][key] += value
        except Exception as e:
            print(f"Error in round {round_num} of {year}: {e}")
            continue
    
    # Calculate averages for relevant statistics
    for team in team_stats:
        if race_count[team] > 0:
            for stat in ['avg_grid_pos', 'avg_finish_pos', 'best_lap_time']:
                if stat in team_stats[team]:
                    team_stats[team][stat] /= race_count[team]
    
    return dict(team_stats)

def normalize_team_name(name):
    """Normalize team names to avoid duplicates"""
    name_mapping = {
        'Alfa Romeo': 'Alfa Romeo',
        'Alfa Romeo Racing': 'Alfa Romeo',
        'AlphaTauri': 'AlphaTauri',
        'Racing Point': 'Aston Martin',
        'Aston Martin': 'Aston Martin',
        'Kick Sauber': 'Sauber',
        'RB': 'RB',
    }
    return name_mapping.get(name, name)

# Create data in long format (one row per team per race)
data = []
print("🏎️  Starting F1 data collection with optimized caching...")

for year in tqdm(YEARS, desc="Processing years"):
    print(f"\n📅 Processing year {year}...")
    
    # First: Get the real final standings from the last race
    champion_team, final_standings = get_final_standings(year)
    if not champion_team:
        print(f"❌ Skipping {year} - could not determine final standings")
        continue
    
    total_rounds_year = TOTAL_ROUNDS.get(year, 24)
    print(f"📊 Season info: {total_rounds_year} total rounds, using rounds {TEST_ROUNDS} for training")
    
    # Second: collect training data for this year
    for round_num in tqdm(TEST_ROUNDS, desc=f"Training rounds for {year}", leave=False):
        print(f"  📊 Analyzing training round {round_num}/{total_rounds_year}...")
        stats = get_accumulated_stats(year, round_num, TEST_ROUNDS)
        if not stats:
            continue
        
        # Create one row per team
        for team_name, metrics in stats.items():
            # Normalize team names to avoid duplicates
            team = normalize_team_name(team_name.strip())
            
            # Calculate how many races were actually loaded for this analysis
            races_analyzed = len([r for r in TEST_ROUNDS if r <= round_num])
            
            # Get final points for this team from the real final standings
            normalized_final_standings = {normalize_team_name(k): v for k, v in final_standings.items()}
            final_team_points = normalized_final_standings.get(team, 0)
            
            entry = {
                # Race information
                "year": year,
                "round": round_num,
                "total_rounds": total_rounds_year,
                "season_progress": round_num / total_rounds_year,
                "team": team,
                "champion": champion_team,
                "is_champion": 1 if team == champion_team else 0,
                "final_points": final_team_points,  # Real final championship points
                
                # Performance metrics (up to this round in training)
                "points": metrics.get('points', 0),
                "avg_grid_position": metrics.get('avg_grid_pos', 20),
                "avg_finish_position": metrics.get('avg_finish_pos', 20),
                "positions_gained": metrics.get('positions_gained', 0),
                "fastest_laps": metrics.get('fastest_laps', 0),
                "best_lap_time": metrics.get('best_lap_time', 0),
                "dnfs": metrics.get('dnfs', 0),
                
                # Derived metrics (based on actual races analyzed)
                "recovery_rate": metrics.get('avg_grid_pos', 20) - metrics.get('avg_finish_pos', 20),
                "points_per_race": metrics.get('points', 0) / races_analyzed if races_analyzed > 0 else 0,
                "reliability_rate": 1 - (metrics.get('dnfs', 0) / (races_analyzed * 2)) if races_analyzed > 0 else 1
            }
            
            data.append(entry)

# Create DataFrame
df = pd.DataFrame(data)

# Define column descriptions for documentation
column_descriptions = {
    'year': 'Year of the F1 season',
    'round': 'Race round number (cumulative data up to this round)',
    'total_rounds': 'Total number of rounds in this season',
    'season_progress': 'Progress through season (0.0 to 1.0)',
    'team': 'Constructor team name',
    'champion': 'Constructor champion of the season',
    'is_champion': 'Binary indicator: 1 if this team is the champion, 0 otherwise',
    'final_points': 'Final championship points for this team (end of season total)',
    'points': 'Total points accumulated by the team up to this round',
    'avg_grid_position': 'Average starting grid position across all races',
    'avg_finish_position': 'Average finishing position across all races',
    'positions_gained': 'Total positions gained/lost from grid to finish',
    'fastest_laps': 'Number of fastest laps achieved',
    'best_lap_time': 'Best lap time in seconds',
    'dnfs': 'Number of Did Not Finish results',
    'recovery_rate': 'Average positions gained/lost from grid to finish (positive = improvement)',
    'points_per_race': 'Average points scored per race',
    'reliability_rate': 'Reliability rate (1 = perfect reliability, 0 = always DNF)'
}

# Handle missing values properly
for col in df.columns:
    if col in ['year', 'round', 'team', 'champion']:
        continue
    elif col == 'best_lap_time':
        # Replace 0s and NaN with mean for lap times
        mean_val = df[col][df[col] > 0].mean()
        df[col] = df[col].replace(0, np.nan).fillna(mean_val)
    elif 'position' in col:
        # Replace 0s with 20 (last position) for positions
        df[col] = df[col].replace(0, 20)
    else:
        # Fill other metrics with 0
        df[col] = df[col].fillna(0)

# Sort data for better analysis
if not df.empty:
    df = df.sort_values(['year', 'round', 'team']).reset_index(drop=True)
else:
    print("⚠️ Nenhum dado foi coletado. Verifique se as rodadas e anos estão corretos.")

# Create metadata DataFrame
metadata_df = pd.DataFrame(
    [(col, desc) for col, desc in column_descriptions.items()],
    columns=['Column Name', 'Description']
)

train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TrainModels'))
os.makedirs(train_dir, exist_ok=True)

# Save the column descriptions
metadata_df.to_csv(os.path.join(train_dir, "ConstructorCVS_metadata.csv"), index=False)

# Create summary statistics
print("✅ Metadata saved in TrainModels directory")
print(f"📊 Dataset shape: {df.shape}")
print(f"🏁 Years covered: {sorted(df['year'].unique())}")
print(f"🏆 Teams included: {sorted(df['team'].unique())}")
print(f"📈 Champions by year: {df[['year', 'champion']].drop_duplicates().to_dict('records')}")

# Show sample of the data
print("\n📋 Sample data:")
print(df.head(10).to_string())

# Create only the essential CSV files for ML
print("\n🔍 Creating essential datasets for machine learning...")

# 1. Main training dataset (enhanced with season context)
ml_features = [
    'year', 'round', 'total_rounds', 'season_progress', 'team', 'is_champion',  # Core identifiers + context
    'final_points',  # Final championship points for reference
    'points', 'avg_grid_position', 'avg_finish_position',  # Main performance
    'recovery_rate', 'points_per_race', 'reliability_rate'  # Derived metrics
]
ml_df = df[ml_features].copy()
ml_df.to_csv(os.path.join(train_dir, "constructor_ml_dataset.csv"), index=False)

# 2. Championship standings (final results per year for validation)
championship_standings = df.groupby(['year', 'team']).agg({
    'points': 'max',  # Maximum accumulated points (final points)
    'is_champion': 'max'  # Champion status
}).reset_index()

# Add final_points column that matches the maximum points (this is the correct final championship points)
championship_standings['final_points'] = championship_standings['points']
championship_standings = championship_standings.sort_values(['year', 'final_points'], ascending=[True, False])
championship_standings['position'] = championship_standings.groupby('year')['final_points'].rank(method='min', ascending=False).astype(int)

# Reorder columns for clarity
championship_standings = championship_standings[['year', 'team', 'final_points', 'points', 'is_champion', 'position']]
championship_standings.to_csv(os.path.join(train_dir, "championship_standings.csv"), index=False)

print("✅ Essential datasets created:")
print("  📊 constructor_ml_dataset.csv - Main training data")
print("  🏆 championship_standings.csv - Final standings for validation")
print(f"\n💾 Total files created: 2 CSV files + 1 metadata file")
print(f"🗂️  Main dataset for ML: constructor_ml_dataset.csv ({ml_df.shape[0]} rows, {ml_df.shape[1]} columns)")

# Substitui o bloco de geração de championship_standings para usar apenas os pontos finais reais
final_standings_list = []
for year in YEARS:
    champion_team, final_standings = get_final_standings(year)
    for team, points in final_standings.items():
        final_standings_list.append({
            'year': year,
            'team': team,
            'final_points': points,
            'is_champion': 1 if team == champion_team else 0
        })
championship_standings = pd.DataFrame(final_standings_list)
championship_standings = championship_standings.sort_values(['year', 'final_points'], ascending=[True, False])
championship_standings['position'] = championship_standings.groupby('year')['final_points'].rank(method='min', ascending=False).astype(int)
championship_standings = championship_standings[['year', 'team', 'final_points', 'is_champion', 'position']]
championship_standings.to_csv(os.path.join(train_dir, "championship_standings.csv"), index=False)

print("✅ Essential datasets created:")
print("  📊 constructor_ml_dataset.csv - Main training data (apenas TEST_ROUNDS)")
print("  🏆 championship_standings.csv - Final standings reais para validação")
print(f"\n💾 Total files created: 2 CSV files + 1 metadata file")
print(f"🗂️  Main dataset para ML: constructor_ml_dataset.csv ({ml_df.shape[0]} rows, {ml_df.shape[1]} columns)")
print(f"🏆 Validation dataset: championship_standings.csv ({championship_standings.shape[0]} rows, {championship_standings.shape[1]} columns)")
