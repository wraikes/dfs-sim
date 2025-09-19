"""Base data processor for DFS sports."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

from ..models.player import Position
from ..models.site import SiteCode
from ..data.llm_newsletter_processor import LLMNewsletterProcessor, NewsletterExtraction


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
            print("📰 No newsletter signals found - using default values")
            return df

        # Apply signals from LLM extraction
        applied_count = 0
        for signal in extraction.players:
            # Find matching players
            name_upper = signal.name.upper()
            mask = df['name'].str.upper().str.contains(name_upper, na=False)
            matched_players = df[mask]

            if len(matched_players) > 0:
                df.loc[mask, 'newsletter_signal'] = signal.signal
                df.loc[mask, 'newsletter_confidence'] = signal.confidence
                df.loc[mask, 'newsletter_reason'] = signal.reason
                df.loc[mask, 'ownership_delta'] = signal.ownership_delta
                df.loc[mask, 'ceiling_delta'] = signal.ceiling_delta
                applied_count += len(matched_players)

        # Count applied signals by type
        targets_count = len(df[df['newsletter_signal'] == 'target'])
        avoids_count = len(df[df['newsletter_signal'] == 'avoid'])

        print(f"📰 Applied newsletter signals:")
        print(f"  🎯 {targets_count} targets")
        print(f"  ⛔ {avoids_count} avoids")
        print(f"  📊 {applied_count} total players affected")

        return df

    def apply_projection_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Base implementation with newsletter signals and salary-ownership correlation."""
        df = df.copy()
        # Always add updated columns - copy originals as default
        df['updated_projection'] = df['projection']
        df['updated_floor'] = df['floor']
        df['updated_ceiling'] = df['ceiling']
        df['updated_ownership'] = df['ownership']

        # Apply newsletter signals (universal across sports)
        newsletter_players = df[df['newsletter_signal'] != 'neutral']
        for idx, row in newsletter_players.iterrows():
            signal = row['newsletter_signal']
            confidence = row['newsletter_confidence']
            ceiling_delta = row['ceiling_delta']
            ownership_delta = row['ownership_delta']

            if signal == 'target':
                # Targets get projection boost based on confidence
                proj_mult = 1.02 + (0.08 * confidence)  # 1.02-1.10x
                df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
                df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult
                # Use LLM-calculated ceiling delta
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * (1 + ceiling_delta)
                # Use LLM-calculated ownership delta
                df.loc[idx, 'updated_ownership'] = max(0.1, row['ownership'] + (ownership_delta * 100))

            elif signal == 'avoid':
                # Avoids get projection reduction based on confidence
                proj_mult = 0.96 - (0.06 * confidence)  # 0.90-0.96x
                df.loc[idx, 'updated_projection'] = row['projection'] * proj_mult
                df.loc[idx, 'updated_floor'] = row['floor'] * proj_mult
                # Use LLM-calculated ceiling delta
                df.loc[idx, 'updated_ceiling'] = row['ceiling'] * (1 + ceiling_delta)
                # Use LLM-calculated ownership delta
                df.loc[idx, 'updated_ownership'] = row['ownership'] + (ownership_delta * 100)

        # Apply bounds validation
        print("   🔒 Applying bounds validation...")
        # Clamp ownership to [0, 100]
        df['updated_ownership'] = np.clip(df['updated_ownership'], 0, 100)

        # Ensure ceiling >= projection
        mask = df['updated_ceiling'] < df['updated_projection']
        df.loc[mask, 'updated_ceiling'] = df.loc[mask, 'updated_projection'] * 1.1

        # Ensure floor <= projection
        mask = df['updated_floor'] > df['updated_projection']
        df.loc[mask, 'updated_floor'] = df.loc[mask, 'updated_projection'] * 0.8

        # Ensure all values are non-negative
        df['updated_projection'] = np.maximum(df['updated_projection'], 0.1)
        df['updated_floor'] = np.maximum(df['updated_floor'], 0)
        df['updated_ceiling'] = np.maximum(df['updated_ceiling'], df['updated_projection'])

        # Apply salary-ownership correlation model (universal for all sports)
        print("   💰 Applying salary-ownership correlation model...")
        df = self._apply_salary_ownership_correlation(df)

        return df

    def _apply_salary_ownership_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply salary-ownership correlation model with 50/50 weighting."""
        df = df.copy()

        # Calculate salary-based ownership expectations
        min_salary = df['salary'].min()
        max_salary = df['salary'].max()
        salary_range = max_salary - min_salary

        if salary_range > 0:
            # Normalize salaries to 0-1 range
            normalized_salary = (df['salary'] - min_salary) / salary_range
            # Convert to ownership expectations (2% to 20% range based on salary)
            salary_based_ownership = 2 + (normalized_salary * 18)
        else:
            # If all salaries are the same, use uniform ownership
            salary_based_ownership = pd.Series([15.0] * len(df), index=df.index)

        # Create blended ownership with 50/50 weighting
        original_ownership = df['updated_ownership']
        blended_ownership = (0.5 * original_ownership) + (0.5 * salary_based_ownership)

        # Ensure bounds [0.1, 100]
        df['salary_ownership'] = np.clip(blended_ownership, 0.1, 100.0)

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

        # Newsletter signals summary
        signals = df['newsletter_signal'].value_counts()
        if len(signals) > 1:
            print(f"\\nNewsletter Signals:")
            for signal, count in signals.items():
                if signal != 'neutral':
                    print(f"  {signal}: {count}")