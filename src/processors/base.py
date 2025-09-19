"""Base data processor for DFS sports."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from loguru import logger

from ..models.player import Position
from ..models.site import SiteCode
from ..data.llm_newsletter_processor import LLMNewsletterProcessor, NewsletterExtraction
from ..config.projection_config import NEWSLETTER_CONFIG, OWNERSHIP_CONFIG, VALIDATION_CONFIG


class BaseDataProcessor(ABC):
    """Abstract base class for sport-specific data processing."""

    def __init__(self, sport: str, pid: str, site: SiteCode = SiteCode.DK):
        self.sport = sport.lower()
        self.pid = pid
        self.site = site
        self.base_path = Path(f"data/{self.sport}/{self.pid}/{self.site.value}")
        self.json_path = self.base_path / 'json'
        self.csv_path = self.base_path / 'csv'

        # Ensure directories exist
        self.csv_path.mkdir(exist_ok=True)

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Load and combine raw data files into standardized format."""
        pass

    @abstractmethod
    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate sport-specific metrics (ITD, variance, etc.)."""
        pass

    @abstractmethod
    def get_position_from_data(self, row: pd.Series) -> Position:
        """Extract position from raw data."""
        pass

    def process_newsletter_with_llm(self) -> Optional[NewsletterExtraction]:
        """Process newsletter using LLM and create structured signals."""
        # Initialize LLM processor
        llm_processor = LLMNewsletterProcessor(sport=self.sport, model="gpt-4o")

        # Look for newsletter files in the /newsletters subdirectory
        newsletter_dir = self.base_path / 'newsletters'
        if not newsletter_dir.exists():
            return None

        # Find newsletter files
        newsletter_files = list(newsletter_dir.glob("*.txt")) + list(newsletter_dir.glob("*.md"))
        if not newsletter_files:
            print("📰 No newsletter files found - creating empty signals")
            return None

        print(f"📰 Processing newsletter: {newsletter_files[0]}")

        try:
            # Process the first newsletter file found
            extraction = llm_processor.process_newsletter_file(
                str(newsletter_files[0]),
                save_json=True  # Auto-save to /json directory
            )

            print(f"📰 Extracted {len(extraction.players)} player signals")
            return extraction

        except Exception as e:
            print(f"⚠️ LLM newsletter processing failed: {e}")
            return None

    def apply_newsletter_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply newsletter signals to player data using LLM processor."""
        # Always initialize newsletter columns
        df['newsletter_signal'] = 'neutral'
        df['newsletter_confidence'] = 0.5
        df['newsletter_reason'] = ''
        df['ownership_delta'] = 0.0
        df['ceiling_delta'] = 0.0

        # Try to process newsletter
        extraction = self.process_newsletter_with_llm()

        if not extraction or not extraction.players:
            logger.info("No newsletter signals found - using default values")
            return df

        # Apply signals from LLM extraction with improved matching
        applied_count = 0
        for signal in extraction.players:
            matched_indices = self._find_matching_players(df, signal.name)

            if matched_indices:
                df.loc[matched_indices, 'newsletter_signal'] = signal.signal
                df.loc[matched_indices, 'newsletter_confidence'] = signal.confidence
                df.loc[matched_indices, 'newsletter_reason'] = signal.reason
                df.loc[matched_indices, 'ownership_delta'] = signal.ownership_delta
                df.loc[matched_indices, 'ceiling_delta'] = signal.ceiling_delta
                applied_count += len(matched_indices)

                logger.info(f"Matched '{signal.name}' to {len(matched_indices)} players: {signal.signal}")
            else:
                logger.warning(f"No match found for newsletter player: '{signal.name}'")

        # Count applied signals by type
        targets_count = len(df[df['newsletter_signal'] == 'target'])
        avoids_count = len(df[df['newsletter_signal'] == 'avoid'])

        print(f"📰 Applied newsletter signals:")
        print(f"  🎯 {targets_count} targets")
        print(f"  ⛔ {avoids_count} avoids")
        print(f"  📊 {applied_count} total players affected")

        return df

    def _find_matching_players(self, df: pd.DataFrame, newsletter_name: str) -> list:
        """Find players matching newsletter name with improved confidence."""
        newsletter_clean = newsletter_name.upper().strip()

        # Try exact match first
        exact_match = df[df['name'].str.upper() == newsletter_clean].index.tolist()
        if exact_match:
            logger.debug(f"Exact match found for '{newsletter_name}'")
            return exact_match

        # Try partial match with whole words
        mask = df['name'].str.upper().str.contains(newsletter_clean, na=False, regex=False)
        partial_matches = df[mask].index.tolist()

        if len(partial_matches) == 1:
            logger.debug(f"Single partial match found for '{newsletter_name}'")
            return partial_matches
        elif len(partial_matches) > 1:
            logger.warning(f"Multiple partial matches for '{newsletter_name}': {df.loc[partial_matches, 'name'].tolist()}")
            return partial_matches

        # Try last name only
        if ' ' in newsletter_clean:
            last_name = newsletter_clean.split()[-1]
            last_name_mask = df['name'].str.upper().str.contains(last_name, na=False, regex=False)
            last_name_matches = df[last_name_mask].index.tolist()

            if len(last_name_matches) == 1:
                logger.debug(f"Last name match found for '{newsletter_name}'")
                return last_name_matches
            elif len(last_name_matches) > 1:
                logger.warning(f"Multiple last name matches for '{newsletter_name}': {df.loc[last_name_matches, 'name'].tolist()}")

        return []

    def _apply_target_adjustments(self, df: pd.DataFrame, idx: int, row: pd.Series) -> None:
        """Apply adjustments for target players."""
        confidence = row['newsletter_confidence']

        # Use config values instead of hardcoded
        proj_mult = (NEWSLETTER_CONFIG.TARGET_BASE_MULTIPLIER +
                    (NEWSLETTER_CONFIG.TARGET_CONFIDENCE_MULTIPLIER * confidence))

        df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
        df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult

        # Use LLM-calculated ceiling delta
        ceiling_delta = row['ceiling_delta']
        df.loc[idx, 'updated_ceiling'] = row['ceiling'] * (1 + ceiling_delta)

        # Use LLM-calculated ownership delta
        ownership_delta = row['ownership_delta']
        new_ownership = max(NEWSLETTER_CONFIG.MIN_OWNERSHIP,
                           row['ownership'] + (ownership_delta * 100))
        df.loc[idx, 'updated_ownership'] = new_ownership

    def _apply_avoid_adjustments(self, df: pd.DataFrame, idx: int, row: pd.Series) -> None:
        """Apply adjustments for avoid players."""
        confidence = row['newsletter_confidence']

        # Use config values instead of hardcoded
        proj_mult = (NEWSLETTER_CONFIG.AVOID_BASE_MULTIPLIER -
                    (NEWSLETTER_CONFIG.AVOID_CONFIDENCE_MULTIPLIER * confidence))

        df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
        df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult

        # Use LLM-calculated ceiling delta
        ceiling_delta = row['ceiling_delta']
        df.loc[idx, 'updated_ceiling'] = row['ceiling'] * (1 + ceiling_delta)

        # Use LLM-calculated ownership delta
        ownership_delta = row['ownership_delta']
        df.loc[idx, 'updated_ownership'] = row['ownership'] + (ownership_delta * 100)

    def _validate_projections(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate projection adjustments are within reasonable bounds."""
        issues = []

        # Check projection bounds
        proj_issues = df[
            (df['updated_projection'] < VALIDATION_CONFIG.MIN_PROJECTION) |
            (df['updated_projection'] > VALIDATION_CONFIG.MAX_PROJECTION)
        ]
        if not proj_issues.empty:
            issues.append(f"Projection out of bounds: {len(proj_issues)} players")
            logger.warning(f"Projection issues: {proj_issues[['name', 'updated_projection']].to_dict('records')}")

        # Check ceiling bounds
        ceiling_issues = df[
            (df['updated_ceiling'] < VALIDATION_CONFIG.MIN_CEILING) |
            (df['updated_ceiling'] > VALIDATION_CONFIG.MAX_CEILING)
        ]
        if not ceiling_issues.empty:
            issues.append(f"Ceiling out of bounds: {len(ceiling_issues)} players")
            logger.warning(f"Ceiling issues: {ceiling_issues[['name', 'updated_ceiling']].to_dict('records')}")

        # Check ownership bounds
        ownership_issues = df[
            (df['updated_ownership'] < NEWSLETTER_CONFIG.MIN_OWNERSHIP) |
            (df['updated_ownership'] > NEWSLETTER_CONFIG.MAX_OWNERSHIP)
        ]
        if not ownership_issues.empty:
            issues.append(f"Ownership out of bounds: {len(ownership_issues)} players")
            logger.warning(f"Ownership issues: {ownership_issues[['name', 'updated_ownership']].to_dict('records')}")

        return len(issues) == 0, issues

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Base implementation with newsletter signals and salary-ownership correlation."""
        df = df.copy()

        # Always add updated columns - copy originals as default
        df['updated_projection'] = df['projection']
        df['updated_floor'] = df['floor']
        df['updated_ceiling'] = df['ceiling']
        df['updated_ownership'] = df['ownership']

        # Apply newsletter signals using separate functions
        newsletter_players = df[df['newsletter_signal'] != 'neutral']
        for idx, row in newsletter_players.iterrows():
            signal = row['newsletter_signal']

            if signal == 'target':
                self._apply_target_adjustments(df, idx, row)
            elif signal == 'avoid':
                self._apply_avoid_adjustments(df, idx, row)

        # Apply ownership bounds
        logger.debug("Applying ownership bounds validation")
        df['updated_ownership'] = np.clip(df['updated_ownership'],
                                        NEWSLETTER_CONFIG.MIN_OWNERSHIP,
                                        NEWSLETTER_CONFIG.MAX_OWNERSHIP)

        # Apply salary-ownership correlation model
        logger.debug("Applying salary-ownership correlation model")
        df = self._apply_salary_ownership_correlation(df)

        # Validate all adjustments
        is_valid, issues = self._validate_projections(df)
        if not is_valid:
            logger.error(f"Validation failed with {len(issues)} issues: {issues}")

        return df

    def _apply_salary_ownership_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply salary-ownership correlation model using config values."""
        df = df.copy()

        # Calculate salary-based ownership expectations
        min_salary = df['salary'].min()
        max_salary = df['salary'].max()
        salary_range = max_salary - min_salary

        if salary_range > 0:
            # Normalize salaries to 0-1 range
            normalized_salary = (df['salary'] - min_salary) / salary_range
            # Convert to ownership expectations using config values
            ownership_range = OWNERSHIP_CONFIG.MAX_SALARY_OWNERSHIP - OWNERSHIP_CONFIG.MIN_SALARY_OWNERSHIP
            salary_based_ownership = OWNERSHIP_CONFIG.MIN_SALARY_OWNERSHIP + (normalized_salary * ownership_range)
        else:
            # If all salaries are the same, use default ownership
            salary_based_ownership = pd.Series([OWNERSHIP_CONFIG.DEFAULT_OWNERSHIP] * len(df), index=df.index)

        # Create blended ownership using config weights
        original_ownership = df['updated_ownership']
        blended_ownership = (
            (OWNERSHIP_CONFIG.ORIGINAL_WEIGHT * original_ownership) +
            (OWNERSHIP_CONFIG.SALARY_WEIGHT * salary_based_ownership)
        )

        # Ensure bounds
        df['salary_ownership'] = np.clip(blended_ownership,
                                       NEWSLETTER_CONFIG.MIN_OWNERSHIP,
                                       NEWSLETTER_CONFIG.MAX_OWNERSHIP)

        return df

    def process_data(self) -> pd.DataFrame:
        """Main processing workflow."""
        print(f"⚙️ Processing {self.sport.upper()} data for PID: {self.pid}")
        print("=" * 60)

        # 1. Load raw data
        print("1️⃣ Loading raw data files...")
        df = self.load_raw_data()

        # 2. Apply newsletter signals
        print("\\n2️⃣ Applying newsletter signals...")
        df = self.apply_newsletter_signals(df)

        # 3. Calculate sport-specific metrics
        print("\\n3️⃣ Calculating sport metrics...")
        df = self.calculate_sport_metrics(df)

        # 4. Apply projection adjustments
        print("\\n4️⃣ Applying projection adjustments...")
        df = self.apply_projection_adjustments(df)

        return df

    def save_processed_data(self, df: pd.DataFrame) -> Path:
        """Save processed data to CSV."""
        output_file = self.csv_path / f"extracted.csv"
        df.to_csv(output_file, index=False)
        print(f"\\n💾 Saved processed data: {output_file}")
        return output_file

    def display_data_summary(self, df: pd.DataFrame):
        """Display summary of processed data."""
        print(f"\\n📊 DATA SUMMARY")
        print("=" * 60)
        print(f"Players: {len(df)}")
        print(f"Total Salary: ${df['salary'].sum():,}")
        print(f"Avg Projection: {df['updated_projection'].mean():.1f} pts")
        print(f"Avg Ownership: {df['updated_ownership'].mean():.1f}%")

        # Newsletter signals summary with reasons
        signals = df['newsletter_signal'].value_counts()
        if len(signals) > 1:
            print(f"\\nNewsletter Signals:")
            for signal, count in signals.items():
                if signal != 'neutral':
                    print(f"  {signal}: {count}")

            # Show basic newsletter signals for reference
            newsletter_players = df[df['newsletter_signal'] != 'neutral']
            if not newsletter_players.empty:
                print(f"\\nNewsletter Players:")
                for _, player in newsletter_players.iterrows():
                    signal_icon = '🎯' if player['newsletter_signal'] == 'target' else '⛔'
                    print(f"  {signal_icon} {player['name']}")