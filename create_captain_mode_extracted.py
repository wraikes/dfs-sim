#!/usr/bin/env python3
"""
Convert captain_mode.csv (LineStar format) to extracted.csv format for Captain Mode contest.
Creates a new file: dk_382_captain_mode_extracted.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime

def convert_captain_mode_to_extracted():
    # Read the captain mode data
    captain_df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/capital_model.csv')

    # Read the extracted format to get column structure
    extracted_df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/extracted.csv', nrows=1)

    # Create new dataframe with extracted.csv column structure
    captain_extracted = pd.DataFrame()

    # Map captain mode columns to extracted format
    captain_extracted['player_id'] = captain_df['DFS_ID']
    captain_extracted['name'] = captain_df['Name']
    captain_extracted['salary'] = captain_df['Salary']
    captain_extracted['projection'] = captain_df['Projected']
    captain_extracted['floor'] = captain_df['Floor']
    captain_extracted['ceiling'] = captain_df['Ceiling']
    captain_extracted['ownership'] = captain_df['ProjOwn']
    captain_extracted['std_dev'] = 25.0  # Default NFL std_dev
    captain_extracted['position'] = captain_df['Position']
    captain_extracted['team'] = captain_df['Team']

    # Derive opponent from team (LAC@LV game)
    captain_extracted['opponent'] = captain_df['Team'].map({'LAC': 'LV', 'LV': 'LAC'})

    # Game info for LAC@LV
    captain_extracted['game_id'] = 'LAC_LV_CAPTAIN'
    captain_extracted['game_info'] = 'LAC@LV Captain Mode'
    captain_extracted['home_team'] = 'LV'

    # Copy other relevant columns
    captain_extracted['agg_proj'] = captain_df['Consensus']
    captain_extracted['confidence'] = 75.0  # Default confidence
    captain_extracted['stars'] = 3.5  # Default stars
    captain_extracted['alert_score'] = captain_df['AlertScore']
    captain_extracted['opponent_rank'] = captain_df['OppRank']
    captain_extracted['points_per_game'] = captain_df['PPG']

    # Time info (Monday Night game)
    captain_extracted['game_timestamp'] = '/Date(1757966400000-0400)/'
    captain_extracted['game_datetime'] = '2025-09-15 20:00:00'  # Monday Night

    # Default/calculated fields
    captain_extracted['targets_per_game'] = 5.0
    captain_extracted['snap_pct'] = 80.0
    captain_extracted['rz_targets'] = 1.0
    captain_extracted['speed_advantage'] = 0.0
    captain_extracted['shadow_coverage'] = False
    captain_extracted['matchup_fppg_allowed'] = 20.0
    captain_extracted['newsletter_signal'] = 'neutral'
    captain_extracted['newsletter_confidence'] = 0.5
    captain_extracted['newsletter_reason'] = ''
    captain_extracted['ownership_delta'] = 0.0
    captain_extracted['ceiling_delta'] = 0.0
    captain_extracted['variance_multiplier'] = 1.0
    captain_extracted['adjusted_std_dev'] = 25.0
    captain_extracted['ceiling_adjustment'] = 1.0
    captain_extracted['synthesized_floor'] = captain_extracted['floor']
    captain_extracted['adjusted_ceiling'] = captain_extracted['ceiling']
    captain_extracted['ceiling_per_dollar'] = captain_extracted['ceiling'] / captain_extracted['salary'] * 1000
    captain_extracted['projection_per_ownership'] = captain_extracted['projection'] / captain_extracted['ownership'].replace(0, 1)
    captain_extracted['salary_vs_projection_ratio'] = captain_extracted['salary'] / captain_extracted['projection'] / 100
    captain_extracted['sharpe_ratio'] = 0.35
    captain_extracted['ceiling_volatility'] = 0.5
    captain_extracted['confidence_weighted_proj'] = captain_extracted['projection'] * 0.8
    captain_extracted['goal_line_value'] = 0.0
    captain_extracted['target_efficiency'] = 0.0
    captain_extracted['qb_volume_indicator'] = 0.0
    captain_extracted['is_home'] = captain_extracted['team'] == 'LV'
    captain_extracted['game_pace_proxy'] = 210.0
    captain_extracted['team_stack_ceiling'] = 200.0
    captain_extracted['defensive_mismatch'] = 8.0
    captain_extracted['ownership_mispricing'] = -30.0
    captain_extracted['signal_strength'] = 0.0
    captain_extracted['contrarian_boost'] = 0.0
    captain_extracted['updated_projection'] = captain_extracted['projection']
    captain_extracted['updated_floor'] = captain_extracted['floor']
    captain_extracted['updated_ceiling'] = captain_extracted['ceiling']
    captain_extracted['updated_ownership'] = captain_extracted['ownership']
    captain_extracted['salary_ownership'] = captain_extracted['salary'] / captain_extracted['ownership'].replace(0, 1) / 100
    captain_extracted['value'] = captain_extracted['projection'] / captain_extracted['salary'] * 1000

    # Filter out zero salary players (bench/inactive)
    captain_extracted = captain_extracted[captain_extracted['salary'] > 200]

    # Sort by salary descending (Captain mode style)
    captain_extracted = captain_extracted.sort_values('salary', ascending=False)

    # Save as new file
    output_path = '/home/wraikes/programming/portfolio/dfs-sim/data/nfl/382/dk/csv/dk_382_captain_mode_extracted.csv'
    captain_extracted.to_csv(output_path, index=False)

    print(f"Created Captain Mode extracted file: {output_path}")
    print(f"Total players: {len(captain_extracted)}")
    print(f"Salary range: ${captain_extracted['salary'].min():,} - ${captain_extracted['salary'].max():,}")
    print("\nTop 10 by salary:")
    print(captain_extracted[['name', 'position', 'team', 'salary', 'projection', 'ownership']].head(10).to_string(index=False))

if __name__ == '__main__':
    convert_captain_mode_to_extracted()