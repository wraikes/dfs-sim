"""MMA-specific data processor for DFS lineup optimization."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .base import BaseDataProcessor
from ..models.player import Position


class MMADataProcessor(BaseDataProcessor):
    """MMA-specific data processor."""

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MMA-specific projection adjustments."""
        # First apply base adjustments (newsletter + salary-ownership)
        df = super().apply_projection_adjustments(df)

        # Then apply MMA-specific derived metric influences
        print("   🥊 Applying MMA-specific adjustments...")

        # 1. Finishing rate influences ceiling (0-4% boost)
        finishing_boost = df['finishing_rate'] > 0
        df.loc[finishing_boost, 'updated_ceiling'] *= (1 + df.loc[finishing_boost, 'finishing_rate'] * 0.04)

        # 2. Style score influences projection (0-2.7% boost for well-rounded fighters)
        style_boost = df['style_score'] > 0
        df.loc[style_boost, 'updated_projection'] *= (1 + df.loc[style_boost, 'style_score'] * 0.0005)

        # 3. Matchup advantage influences all metrics (0-2% boost)
        positive_matchup = df['matchup_advantage'] > 0
        df.loc[positive_matchup, 'updated_projection'] *= (1 + df.loc[positive_matchup, 'matchup_advantage'] * 0.002)
        df.loc[positive_matchup, 'updated_floor'] *= (1 + df.loc[positive_matchup, 'matchup_advantage'] * 0.002)
        df.loc[positive_matchup, 'updated_ceiling'] *= (1 + df.loc[positive_matchup, 'matchup_advantage'] * 0.002)

        # 4. Takedown effectiveness influences floor (0-1.3% boost for strong grapplers)
        takedown_eff = df['takedowns_per_fight'] * (1 - df['takedown_defense'] / 100)
        strong_grapplers = takedown_eff > 0.5
        df.loc[strong_grapplers, 'updated_floor'] *= (1 + (takedown_eff.loc[strong_grapplers] - 0.5) * 0.027)

        # 5. Vegas odds adjustments (most important data source)
        print("   🎰 Applying Vegas odds adjustments...")

        for idx, row in df.iterrows():
            ml_odds = row['ml_odds']
            if ml_odds == 0:  # Skip if no odds data
                continue

            # Convert ML odds to win probability
            if ml_odds < 0:  # Favorite
                win_prob = abs(ml_odds) / (abs(ml_odds) + 100)
                # Higher floor for favorites (win more often)
                floor_boost = win_prob * 0.15  # Up to 15% boost for heavy favorites
                df.loc[idx, 'updated_floor'] *= (1 + floor_boost)
            else:  # Dog
                win_prob = 100 / (ml_odds + 100)
                # Higher ceiling for dogs (upset potential)
                ceiling_boost = (1 - win_prob) * 0.20  # Up to 20% boost for heavy dogs
                df.loc[idx, 'updated_ceiling'] *= (1 + ceiling_boost)

        # Apply takedown matchup ceiling adjustments (MMA-specific)
        print("   🤼 Applying takedown matchup ceiling adjustments...")
        df = self._apply_takedown_ceiling_adjustments(df)

        # Final bounds validation after MMA-specific adjustments
        df['updated_ownership'] = np.clip(df['updated_ownership'], 0, 100)
        mask = df['updated_ceiling'] < df['updated_projection']
        df.loc[mask, 'updated_ceiling'] = df.loc[mask, 'updated_projection'] * 1.1
        mask = df['updated_floor'] > df['updated_projection']
        df.loc[mask, 'updated_floor'] = df.loc[mask, 'updated_projection'] * 0.8

        return df

    def _apply_takedown_ceiling_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ceiling adjustments based on takedown matchup disadvantages."""
        df = df.copy()

        for idx, row in df.iterrows():
            takedown_matchup = row['takedown_matchup']

            # If opponent will dominate takedowns (high score), suppress ceiling
            if takedown_matchup > 5.0:  # Opponent gets 5+ takedowns vs player's defense
                # Severe ceiling suppression for terrible matchups
                ceiling_reduction = min(0.40, (takedown_matchup - 5) * 0.08)  # Up to 40% reduction
                df.loc[idx, 'updated_ceiling'] *= (1 - ceiling_reduction)

            elif takedown_matchup > 3.0:  # Moderate takedown disadvantage
                # Moderate ceiling suppression
                ceiling_reduction = min(0.25, (takedown_matchup - 3) * 0.10)  # Up to 25% reduction
                df.loc[idx, 'updated_ceiling'] *= (1 - ceiling_reduction)

            elif takedown_matchup > 1.5:  # Slight takedown disadvantage
                # Minor ceiling suppression
                ceiling_reduction = min(0.15, (takedown_matchup - 1.5) * 0.08)  # Up to 15% reduction
                df.loc[idx, 'updated_ceiling'] *= (1 - ceiling_reduction)

        # Display notable adjustments
        severe_disadvantages = df[df['takedown_matchup'] > 5.0]
        if not severe_disadvantages.empty:
            print(f"      🚨 Severe takedown disadvantages: {len(severe_disadvantages)} fighters")
            for _, fighter in severe_disadvantages.iterrows():
                print(f"         {fighter['name']}: {fighter['takedown_matchup']:.1f} TD matchup")

        return df

    def _calculate_takedown_matchup(self, df: pd.DataFrame) -> pd.Series:
        """Calculate proper opponent-based takedown matchup scores."""
        takedown_scores = []

        for _, row in df.iterrows():
            player_td_defense = row['takedown_defense']
            opponent_team = row.get('oteam', '')  # Use opponent team field

            if opponent_team:
                # Find opponent using team-based matching
                opponent_row = df[df['hteam'] == opponent_team]
                if not opponent_row.empty:
                    opp_td_per_fight = opponent_row.iloc[0]['takedowns_per_fight']

                    # Calculate actual matchup: opponent TD rate vs player TD defense
                    # Higher score = opponent will dominate takedowns (bad for player ceiling)
                    matchup_score = opp_td_per_fight / (player_td_defense / 100 + 0.1)
                    takedown_scores.append(matchup_score)
                else:
                    # No opponent found, use self-referential as fallback
                    takedown_scores.append(row['takedowns_per_fight'] / (100 - player_td_defense + 1))
            else:
                # No opponent team listed, use self-referential as fallback
                takedown_scores.append(row['takedowns_per_fight'] / (100 - player_td_defense + 1))

        return pd.Series(takedown_scores, index=df.index)

    def load_raw_data(self) -> pd.DataFrame:
        """Load and parse MMA JSON data into standardized CSV format."""
        # Look for raw.json file in json directory
        main_json_file = self.json_path / "raw.json"

        if not main_json_file.exists():
            raise FileNotFoundError(f"No raw.json file found in {self.json_path}")

        print(f"   📄 Loading JSON: {main_json_file}")

        with open(main_json_file, 'r') as f:
            data = json.load(f)

        # Extract salary data from SalaryContainerJson (MMA-specific parsing)
        try:
            salary_data = json.loads(data['SalaryContainerJson'])

            # Extract ownership data from Ownership section
            ownership_map = {}
            if 'Ownership' in data and 'Projected' in data['Ownership']:
                projected = data['Ownership']['Projected']
                # Get the contest ID (should be one key in Projected)
                contest_ids = list(projected.keys())
                if contest_ids:
                    contest_id = contest_ids[0]
                    for ownership_player in projected[contest_id]:
                        salary_id = ownership_player.get('SalaryId')
                        owned_pct = ownership_player.get('Owned', 10.0)
                        if salary_id:
                            ownership_map[salary_id] = owned_pct

            players = []

            for player_data in salary_data['Salaries']:
                player_id = player_data['PID']
                salary_id = player_data['Id']  # Used to match with ownership

                # Get ownership from ownership_map using SalaryId
                ownership = ownership_map.get(salary_id, 10.0)

                player = {
                    'player_id': player_id,
                    'name': player_data['Name'],
                    'salary': player_data['SAL'],
                    'projection': player_data.get('PP', 0.0),  # PP = Projected Points
                    'floor': player_data.get('Floor', 0.0),
                    'ceiling': player_data.get('Ceil', 0.0),
                    'ownership': ownership,
                    'std_dev': 25.0,  # Default variance for MMA
                    'hteam': player_data.get('HTEAM', ''),  # Home team
                    'oteam': player_data.get('OTEAM', ''),  # Opponent team
                    'htid': player_data.get('HTID', 0),  # Home team ID
                    'otid': player_data.get('OTID', 0),  # Opponent team ID
                    'agg_proj': player_data.get('AggProj', 0.0),  # Aggregate projection
                    'confidence': player_data.get('Conf', 50),  # Projection confidence
                    'game_info': player_data.get('GI', ''),  # Game info with fight time
                    'opp_rank': player_data.get('OppRank', 0),  # Opponent ranking
                    'stars': player_data.get('Stars', 3),  # Player star rating
                    'alert_score': player_data.get('AlertScore', 0),  # Alert/attention score
                }
                players.append(player)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ❌ Error parsing SalaryContainerJson: {e}")
            raise ValueError(f"Invalid LineStar data format: {e}")

        # Create base dataframe
        df = pd.DataFrame(players)

        # Extract ML odds and additional data from MatchupData (MMA-specific)
        ml_odds_map = {}
        win_pct_map = {}
        itd_odds_map = {}

        for table in data.get('MatchupData', []):
            if table.get('Name') == 'Fight':
                columns = table['Columns']
                for match in table.get('PlayerMatchups', []):
                    player_id = match['PlayerId']
                    values = match['Values']

                    # Map columns to values
                    col_data = dict(zip(columns, values))

                    # Extract ML odds (Vegas Odds column)
                    if 'Vegas Odds' in col_data:
                        ml_odds_map[player_id] = float(col_data['Vegas Odds']) if col_data['Vegas Odds'] != '0' else 0

                    # Extract win percentage
                    if 'Win Pct' in col_data:
                        win_pct_map[player_id] = float(col_data['Win Pct']) if col_data['Win Pct'] != '0' else 0

                    # Extract full fight odds (ITD odds)
                    if 'Full Fight Odds' in col_data:
                        itd_odds_map[player_id] = float(col_data['Full Fight Odds']) if col_data['Full Fight Odds'] != '0' else 0

        # Add ML odds and win data to dataframe
        df['ml_odds'] = df['player_id'].map(ml_odds_map).fillna(0)
        df['win_pct'] = df['player_id'].map(win_pct_map).fillna(0)
        df['itd_odds'] = df['player_id'].map(itd_odds_map).fillna(0)

        # Extract opponent relationships using OTEAM-to-lastname matching
        opponent_map = {}
        opponent_odds_map = {}

        # Create lastname-to-fighter mapping for OTEAM matching
        lastname_to_fighter = {}
        for _, row in df.iterrows():
            # Extract last name (handle multi-part names like "Jose Daniel Medina")
            last_name = row['name'].split()[-1].upper()
            lastname_to_fighter[last_name] = {
                'player_id': row['player_id'],
                'name': row['name'],
                'ml_odds': row['ml_odds']
            }

        # Match fighters to opponents using OTEAM values
        matched_fighters = set()
        for _, row in df.iterrows():
            player_id = row['player_id']
            player_oteam = row['oteam']

            # Skip if already matched or no OTEAM data
            if player_id in matched_fighters or not player_oteam:
                continue

            # Find opponent by matching OTEAM to fighter lastname
            # Special handling for multi-word names like "DANIEL MEDINA"
            potential_opponents = []

            # First try exact OTEAM match
            if player_oteam in lastname_to_fighter:
                potential_opponents.append(lastname_to_fighter[player_oteam])

            # For multi-word OTEAM like "DANIEL MEDINA", try to find "Jose Daniel Medina"
            elif ' ' in player_oteam:
                for fighter_name, fighter_data in lastname_to_fighter.items():
                    full_fighter_name = fighter_data['name'].upper()
                    if player_oteam in full_fighter_name:
                        potential_opponents.append(fighter_data)

            # Match with the first valid opponent found
            for opponent_data in potential_opponents:
                opponent_id = opponent_data['player_id']

                # Don't match a fighter to themselves and ensure unique matching
                if opponent_id != player_id and opponent_id not in matched_fighters:
                    opponent_map[player_id] = opponent_data['name']
                    opponent_odds_map[player_id] = opponent_data['ml_odds']
                    opponent_map[opponent_id] = row['name']
                    opponent_odds_map[opponent_id] = row['ml_odds']

                    matched_fighters.add(player_id)
                    matched_fighters.add(opponent_id)
                    break

        # Add opponent data to dataframe
        df['opponent'] = df['player_id'].map(opponent_map).fillna('')
        df['opponent_ml_odds'] = df['player_id'].map(opponent_odds_map).fillna(0)

        print(f"   ✅ Parsed {len(df)} fighters from JSON")
        print(f"   🥊 Found {len(opponent_map)} opponent relationships using OTEAM/HTEAM data")
        return df

    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MMA-specific metrics and variability features."""
        df = df.copy()

        # Load the main JSON data to access MatchupData
        main_json_file = self.json_path / "raw.json"
        with open(main_json_file, 'r') as f:
            data = json.load(f)

        # Initialize feature maps for variability analysis
        feature_maps = {
            # Last 5 Fights offensive stats
            'strikes_per_fight': {},
            'strikes_per_min': {},
            'strike_accuracy': {},
            'control_per_fight': {},
            'takedowns_per_fight': {},
            'avg_rounds': {},
            'last5_win_pct': {},
            'last5_fppg': {},

            # Last 5 Fights defensive stats
            'strikes_absorbed': {},
            'strike_defense': {},
            'takedown_defense': {},
            'fppg_allowed': {},

            # Opponent stats for matchup analysis
            'opp_strikes_per_fight': {},
            'opp_strike_accuracy': {},
            'opp_last5_fppg': {},
            'opp_takedowns_per_fight': {},
        }

        # Process MatchupData tables for detailed stats
        for table in data.get('MatchupData', []):
            table_name = table.get('Name', '').lower()
            columns = table.get('Columns', [])

            for match in table.get('PlayerMatchups', []):
                player_id = match['PlayerId']
                values = match['Values']
                col_data = dict(zip(columns, values))

                # Extract Last 5 Fights offensive stats
                if 'last 5 fights (off)' in table_name:
                    if 'SS Landed/F' in col_data:
                        feature_maps['strikes_per_fight'][player_id] = float(col_data['SS Landed/F']) if col_data['SS Landed/F'] != '0' else 0
                    if 'SS Landed/Min' in col_data:
                        feature_maps['strikes_per_min'][player_id] = float(col_data['SS Landed/Min']) if col_data['SS Landed/Min'] != '0' else 0
                    if 'Strike Acc%' in col_data:
                        feature_maps['strike_accuracy'][player_id] = float(col_data['Strike Acc%']) if col_data['Strike Acc%'] != '0' else 0
                    if 'Ctrl Secs/F' in col_data:
                        feature_maps['control_per_fight'][player_id] = float(col_data['Ctrl Secs/F']) if col_data['Ctrl Secs/F'] != '0' else 0
                    if 'Takedowns/F' in col_data:
                        feature_maps['takedowns_per_fight'][player_id] = float(col_data['Takedowns/F']) if col_data['Takedowns/F'] != '0' else 0
                    if 'Rounds/F' in col_data:
                        feature_maps['avg_rounds'][player_id] = float(col_data['Rounds/F']) if col_data['Rounds/F'] != '0' else 0
                    if 'Win%' in col_data:
                        feature_maps['last5_win_pct'][player_id] = float(col_data['Win%']) if col_data['Win%'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['last5_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0

                # Extract Last 5 Fights defensive stats
                elif 'last 5 fights (def)' in table_name:
                    if 'SS Taken/F' in col_data:
                        feature_maps['strikes_absorbed'][player_id] = float(col_data['SS Taken/F']) if col_data['SS Taken/F'] != '0' else 0
                    if 'Strike Def%' in col_data:
                        feature_maps['strike_defense'][player_id] = float(col_data['Strike Def%']) if col_data['Strike Def%'] != '0' else 0
                    if 'Takedown Def%' in col_data:
                        feature_maps['takedown_defense'][player_id] = float(col_data['Takedown Def%']) if col_data['Takedown Def%'] != '0' else 0
                    if 'FPPG Allowed' in col_data:
                        feature_maps['fppg_allowed'][player_id] = float(col_data['FPPG Allowed']) if col_data['FPPG Allowed'] != '0' else 0

                # Extract opponent stats for matchup analysis
                elif 'opp last 5 (off)' in table_name:
                    if 'SS Landed/F' in col_data:
                        feature_maps['opp_strikes_per_fight'][player_id] = float(col_data['SS Landed/F']) if col_data['SS Landed/F'] != '0' else 0
                    if 'Strike Acc%' in col_data:
                        feature_maps['opp_strike_accuracy'][player_id] = float(col_data['Strike Acc%']) if col_data['Strike Acc%'] != '0' else 0
                    if 'FPPG' in col_data:
                        feature_maps['opp_last5_fppg'][player_id] = float(col_data['FPPG']) if col_data['FPPG'] != '0' else 0
                    if 'Takedowns/F' in col_data:
                        feature_maps['opp_takedowns_per_fight'][player_id] = float(col_data['Takedowns/F']) if col_data['Takedowns/F'] != '0' else 0

        # Add all features to dataframe
        for feature_name, feature_map in feature_maps.items():
            df[feature_name] = df['player_id'].map(feature_map).fillna(0)

        # Calculate derived metrics for variability analysis
        df['finishing_rate'] = df.apply(lambda row: max(0, 3 - row['avg_rounds']) / 3 if row['avg_rounds'] > 0 else 0.33, axis=1)
        df['style_score'] = df['strikes_per_min'] / (df['takedowns_per_fight'] + 0.1)  # Striker vs grappler indicator
        df['matchup_advantage'] = df['strike_accuracy'] - df['opp_strike_accuracy']  # Striking matchup edge
        # Calculate proper opponent-based takedown matchup
        df['takedown_matchup'] = self._calculate_takedown_matchup(df)

        # Calculate ITD probability from ML odds and finishing rate
        df['itd_probability'] = 0.35  # Default
        if 'ml_odds' in df.columns:
            # Combine ML odds with finishing rate for better ITD estimates
            df.loc[df['ml_odds'] <= -200, 'itd_probability'] = 0.45 + df.loc[df['ml_odds'] <= -200, 'finishing_rate'] * 0.15
            df.loc[df['ml_odds'] >= 200, 'itd_probability'] = 0.25 + df.loc[df['ml_odds'] >= 200, 'finishing_rate'] * 0.1

        # Calculate ITD-adjusted ceiling
        df['itd_adjusted_ceiling'] = df.apply(
            lambda row: row['ceiling'] * (1 + 0.4 * row['itd_probability']), axis=1
        )


        print(f"   ✅ Added {len(feature_maps)} variability features to {len(df)} fighters")

        return df

    def get_position_from_data(self, _row: pd.Series) -> Position:
        """MMA fighters are all FIGHTER position."""
        return Position.FIGHTER