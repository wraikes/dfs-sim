#!/usr/bin/env python3
"""
Enhance captain_mode.csv by adding all the advanced features from extracted.csv.
Preserves original Captain Mode data and pricing while adding calculated features.
"""

import pandas as pd
import numpy as np

def enhance_captain_mode_with_features():
    # Read the captain mode data (base data to preserve)
    captain_df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/capital_model.csv')

    # Read extracted.csv to get the column structure we need to match
    extracted_df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/extracted.csv', nrows=1)

    # Create enhanced dataframe starting with captain mode data
    enhanced = captain_df.copy()

    # Rename/map existing columns to match extracted.csv naming
    column_mapping = {
        'LineStarId': 'player_id',
        'Name': 'name',
        'Position': 'position',
        'Team': 'team',
        'Salary': 'salary',
        'Projected': 'projection',
        'Floor': 'floor',
        'Ceiling': 'ceiling',
        'ProjOwn': 'ownership'
    }

    enhanced = enhanced.rename(columns=column_mapping)

    # Add missing core columns
    enhanced['opponent'] = enhanced['team'].map({'LAC': 'LV', 'LV': 'LAC'})
    enhanced['std_dev'] = 25.0  # Standard NFL variance
    enhanced['game_id'] = 'LAC_LV_CAPTAIN'
    enhanced['game_info'] = 'LAC@LV Captain Mode'
    enhanced['home_team'] = 'LV'
    enhanced['agg_proj'] = enhanced.get('Consensus', enhanced['projection'])
    enhanced['confidence'] = 75.0
    enhanced['stars'] = 3.5
    enhanced['alert_score'] = enhanced.get('AlertScore', 0)
    enhanced['opponent_rank'] = enhanced.get('OppRank', 15)
    enhanced['points_per_game'] = enhanced.get('PPG', enhanced['projection'])

    # Time columns
    enhanced['game_timestamp'] = '/Date(1757966400000-0400)/'  # Monday night
    enhanced['game_datetime'] = '2025-09-15 20:00:00'

    # Player usage metrics
    enhanced['targets_per_game'] = np.where(enhanced['position'].isin(['WR', 'TE']), 5.0, 1.0)
    enhanced['snap_pct'] = np.where(enhanced['position'] == 'QB', 95.0,
                           np.where(enhanced['position'].isin(['RB', 'WR', 'TE']), 70.0, 100.0))
    enhanced['rz_targets'] = np.where(enhanced['position'].isin(['WR', 'TE']), 1.0, 0.0)
    enhanced['speed_advantage'] = 0.0
    enhanced['shadow_coverage'] = False
    enhanced['matchup_fppg_allowed'] = 20.0

    # Newsletter signal columns
    enhanced['newsletter_signal'] = 'neutral'
    enhanced['newsletter_confidence'] = 0.5
    enhanced['newsletter_reason'] = ''
    enhanced['ownership_delta'] = 0.0
    enhanced['ceiling_delta'] = 0.0
    enhanced['variance_multiplier'] = 1.0
    enhanced['adjusted_std_dev'] = enhanced['std_dev']
    enhanced['ceiling_adjustment'] = 1.0
    enhanced['synthesized_floor'] = enhanced['floor']
    enhanced['adjusted_ceiling'] = enhanced['ceiling']

    # Calculated ratio columns
    enhanced['ceiling_per_dollar'] = enhanced['ceiling'] / enhanced['salary'] * 1000
    enhanced['projection_per_ownership'] = enhanced['projection'] / enhanced['ownership'].replace(0, 0.1)  # Avoid div by 0
    enhanced['salary_vs_projection_ratio'] = enhanced['salary'] / enhanced['projection'] / 100
    enhanced['sharpe_ratio'] = 0.35
    enhanced['ceiling_volatility'] = enhanced['ceiling'] / enhanced['projection'] - 1
    enhanced['confidence_weighted_proj'] = enhanced['projection'] * (enhanced['confidence'] / 100)

    # Game script columns
    enhanced['goal_line_value'] = np.where(enhanced['position'] == 'RB', 0.5, 0.0)
    enhanced['target_efficiency'] = np.where(enhanced['position'].isin(['WR', 'TE']),
                                    enhanced['targets_per_game'] * 1.5, 0.0)
    enhanced['qb_volume_indicator'] = np.where(enhanced['position'] == 'QB',
                                      enhanced['projection'] / 20.0, 0.0)
    enhanced['is_home'] = (enhanced['team'] == 'LV').astype(int)
    enhanced['game_pace_proxy'] = 210.0  # Captain mode typically higher pace
    enhanced['team_stack_ceiling'] = enhanced.groupby('team')['ceiling'].transform('sum')
    enhanced['defensive_mismatch'] = 8.0
    enhanced['ownership_mispricing'] = enhanced['salary'] / enhanced['ownership'] - 500  # Captain mode premium
    enhanced['signal_strength'] = 0.0
    enhanced['contrarian_boost'] = np.where(enhanced['ownership'] < 5, 1.2, 0.0)

    # Updated columns (same as base for now)
    enhanced['updated_projection'] = enhanced['projection']
    enhanced['updated_floor'] = enhanced['floor']
    enhanced['updated_ceiling'] = enhanced['ceiling']
    enhanced['updated_ownership'] = enhanced['ownership']

    # Final calculated columns
    enhanced['salary_ownership'] = enhanced['salary'] / enhanced['ownership'].replace(0, 0.1) / 100
    enhanced['value'] = enhanced['projection'] / enhanced['salary'] * 1000

    # Get the exact column order from extracted.csv
    target_columns = extracted_df.columns.tolist()

    # Ensure all target columns exist (fill missing with defaults)
    for col in target_columns:
        if col not in enhanced.columns:
            enhanced[col] = 0.0  # Default for any missing columns

    # Reorder columns to match extracted.csv exactly
    enhanced = enhanced[target_columns]

    # Filter out very low salary players (likely inactive)
    enhanced = enhanced[enhanced['salary'] >= 200]

    # Sort by salary descending (Captain mode style)
    enhanced = enhanced.sort_values('salary', ascending=False)

    # Save enhanced captain mode file
    output_path = '/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/dk_382_captain_mode_extracted.csv'
    enhanced.to_csv(output_path, index=False)

    print(f"Enhanced Captain Mode file created: {output_path}")
    print(f"Total players: {len(enhanced)}")
    print(f"Salary range: ${enhanced['salary'].min():,} - ${enhanced['salary'].max():,}")
    print(f"Total columns: {len(enhanced.columns)}")
    print("\nTop 10 by salary:")
    print(enhanced[['name', 'position', 'team', 'salary', 'projection', 'ownership']].head(10).to_string(index=False))

if __name__ == '__main__':
    enhance_captain_mode_with_features()