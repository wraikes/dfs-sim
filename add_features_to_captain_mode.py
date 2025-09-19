#!/usr/bin/env python3
"""
Add extracted.csv features to captain_mode.csv without losing any original columns.
Purely additive approach - keeps all captain_mode columns and adds new ones.
"""

import pandas as pd
import numpy as np

def add_features_to_captain_mode():
    # Read both files
    captain_df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/capital_model.csv')
    extracted_df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/extracted.csv', nrows=1)

    # Start with captain_mode.csv as base (preserve everything)
    enhanced = captain_df.copy()

    # Get columns that exist in extracted but not in captain_mode
    captain_columns = set(captain_df.columns)
    extracted_columns = set(extracted_df.columns)
    new_columns_to_add = extracted_columns - captain_columns

    print(f"Captain mode has {len(captain_columns)} columns")
    print(f"Extracted has {len(extracted_columns)} columns")
    print(f"Adding {len(new_columns_to_add)} new columns")
    print(f"New columns: {sorted(new_columns_to_add)}")

    # Add missing columns with calculated values
    for col in new_columns_to_add:
        if col == 'player_id':
            enhanced[col] = enhanced['DFS_ID']
        elif col == 'name':
            enhanced[col] = enhanced['Name']
        elif col == 'salary':
            enhanced[col] = enhanced['Salary']
        elif col == 'projection':
            enhanced[col] = enhanced['Projected']
        elif col == 'floor':
            enhanced[col] = enhanced['Floor']
        elif col == 'ceiling':
            enhanced[col] = enhanced['Ceiling']
        elif col == 'ownership':
            enhanced[col] = enhanced['ProjOwn']
        elif col == 'position':
            enhanced[col] = enhanced['Position']
        elif col == 'team':
            enhanced[col] = enhanced['Team']
        elif col == 'opponent':
            enhanced[col] = enhanced['Team'].map({'LAC': 'LV', 'LV': 'LAC'})
        elif col == 'std_dev':
            enhanced[col] = 25.0
        elif col == 'game_id':
            enhanced[col] = 'LAC_LV_CAPTAIN'
        elif col == 'game_info':
            enhanced[col] = 'LAC@LV Captain Mode'
        elif col == 'home_team':
            enhanced[col] = 'LV'
        elif col == 'agg_proj':
            enhanced[col] = enhanced.get('Consensus', enhanced['Projected'])
        elif col == 'confidence':
            enhanced[col] = 75.0
        elif col == 'stars':
            enhanced[col] = 3.5
        elif col == 'alert_score':
            enhanced[col] = enhanced.get('AlertScore', 0)
        elif col == 'opponent_rank':
            enhanced[col] = enhanced.get('OppRank', 15)
        elif col == 'points_per_game':
            enhanced[col] = enhanced.get('PPG', enhanced['Projected'])
        elif col == 'game_timestamp':
            enhanced[col] = '/Date(1757966400000-0400)/'
        elif col == 'game_datetime':
            enhanced[col] = '2025-09-15 20:00:00'
        elif col == 'targets_per_game':
            enhanced[col] = np.where(enhanced['Position'].isin(['WR', 'TE']), 5.0, 1.0)
        elif col == 'snap_pct':
            enhanced[col] = np.where(enhanced['Position'] == 'QB', 95.0,
                            np.where(enhanced['Position'].isin(['RB', 'WR', 'TE']), 70.0, 100.0))
        elif col == 'rz_targets':
            enhanced[col] = np.where(enhanced['Position'].isin(['WR', 'TE']), 1.0, 0.0)
        elif col == 'speed_advantage':
            enhanced[col] = 0.0
        elif col == 'shadow_coverage':
            enhanced[col] = False
        elif col == 'matchup_fppg_allowed':
            enhanced[col] = 20.0
        elif col == 'newsletter_signal':
            enhanced[col] = 'neutral'
        elif col == 'newsletter_confidence':
            enhanced[col] = 0.5
        elif col == 'newsletter_reason':
            enhanced[col] = ''
        elif col == 'ownership_delta':
            enhanced[col] = 0.0
        elif col == 'ceiling_delta':
            enhanced[col] = 0.0
        elif col == 'variance_multiplier':
            enhanced[col] = 1.0
        elif col == 'adjusted_std_dev':
            enhanced[col] = 25.0
        elif col == 'ceiling_adjustment':
            enhanced[col] = 1.0
        elif col == 'synthesized_floor':
            enhanced[col] = enhanced['Floor']
        elif col == 'adjusted_ceiling':
            enhanced[col] = enhanced['Ceiling']
        elif col == 'ceiling_per_dollar':
            enhanced[col] = enhanced['Ceiling'] / enhanced['Salary'] * 1000
        elif col == 'projection_per_ownership':
            enhanced[col] = enhanced['Projected'] / enhanced['ProjOwn'].replace(0, 0.1)
        elif col == 'salary_vs_projection_ratio':
            enhanced[col] = enhanced['Salary'] / enhanced['Projected'] / 100
        elif col == 'sharpe_ratio':
            enhanced[col] = 0.35
        elif col == 'ceiling_volatility':
            enhanced[col] = enhanced['Ceiling'] / enhanced['Projected'] - 1
        elif col == 'confidence_weighted_proj':
            enhanced[col] = enhanced['Projected'] * 0.75
        elif col == 'goal_line_value':
            enhanced[col] = np.where(enhanced['Position'] == 'RB', 0.5, 0.0)
        elif col == 'target_efficiency':
            enhanced[col] = np.where(enhanced['Position'].isin(['WR', 'TE']), 5.0 * 1.5, 0.0)
        elif col == 'qb_volume_indicator':
            enhanced[col] = np.where(enhanced['Position'] == 'QB', enhanced['Projected'] / 20.0, 0.0)
        elif col == 'is_home':
            enhanced[col] = (enhanced['Team'] == 'LV').astype(int)
        elif col == 'game_pace_proxy':
            enhanced[col] = 210.0
        elif col == 'team_stack_ceiling':
            enhanced[col] = enhanced.groupby('Team')['Ceiling'].transform('sum')
        elif col == 'defensive_mismatch':
            enhanced[col] = 8.0
        elif col == 'ownership_mispricing':
            enhanced[col] = enhanced['Salary'] / enhanced['ProjOwn'].replace(0, 0.1) - 500
        elif col == 'signal_strength':
            enhanced[col] = 0.0
        elif col == 'contrarian_boost':
            enhanced[col] = np.where(enhanced['ProjOwn'] < 5, 1.2, 0.0)
        elif col == 'updated_projection':
            enhanced[col] = enhanced['Projected']
        elif col == 'updated_floor':
            enhanced[col] = enhanced['Floor']
        elif col == 'updated_ceiling':
            enhanced[col] = enhanced['Ceiling']
        elif col == 'updated_ownership':
            enhanced[col] = enhanced['ProjOwn']
        elif col == 'salary_ownership':
            enhanced[col] = enhanced['Salary'] / enhanced['ProjOwn'].replace(0, 0.1) / 100
        elif col == 'value':
            enhanced[col] = enhanced['Projected'] / enhanced['Salary'] * 1000
        else:
            # Default value for any other missing columns
            enhanced[col] = 0.0

    # Save enhanced file
    output_path = '/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/dk_382_captain_mode_extracted.csv'
    enhanced.to_csv(output_path, index=False)

    print(f"\nEnhanced Captain Mode file created: {output_path}")
    print(f"Total players: {len(enhanced)}")
    print(f"Total columns: {len(enhanced.columns)} (was {len(captain_df.columns)})")
    print(f"Salary range: ${enhanced['Salary'].min():,} - ${enhanced['Salary'].max():,}")
    print("\nTop 5 by salary:")
    print(enhanced[['Name', 'Position', 'Team', 'Salary', 'Projected', 'ProjOwn']].head().to_string(index=False))

if __name__ == '__main__':
    add_features_to_captain_mode()