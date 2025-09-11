#!/usr/bin/env python3
"""
DFS Simulator Pipeline
End-to-end pipeline for DFS lineup optimization
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
import click
from dotenv import load_dotenv

from src.ingestion.data_fetcher import DataFetcher
from src.ingestion.json_to_csv import JSONToCSVConverter
from src.adjustments.newsletter_parser import NewsletterParser
from src.adjustments.projection_modifier import ProjectionModifier
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.correlation_builder import CorrelationBuilder
from src.optimization.lineup_optimizer import LineupOptimizer
from src.optimization.lineup_selector import LineupSelector
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class DFSPipeline:
    """Main pipeline orchestrator for DFS optimization"""
    
    def __init__(self, site: str, sport: str, slate_id: Optional[str] = None):
        self.site = site.lower()
        self.sport = sport.lower()
        self.slate_id = slate_id or datetime.now().strftime("%Y%m%d")
        self.data_dir = Path("data")
        self.output_dir = Path("outputs")
        
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_path in [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "cache",
            self.output_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Execute the full pipeline"""
        logger.info(f"Starting DFS Pipeline for {self.site} - {self.sport}")
        
        # Step 1: Provide data source link
        print("\n" + "="*60)
        print("STEP 1: DATA COLLECTION")
        print("="*60)
        fetcher = DataFetcher(self.site, self.sport)
        data_url = fetcher.get_data_url()
        json_path = self.data_dir / "raw" / f"{self.site}_{self.sport}_{self.slate_id}.json"
        
        print(f"\nPlease visit this URL and copy the data:")
        print(f"🔗 {data_url}")
        print(f"\nSave the JSON data to: {json_path}")
        input("\nPress Enter when you've saved the file...")
        
        # Step 2: Verify file exists
        if not json_path.exists():
            logger.error(f"File not found: {json_path}")
            return
        
        # Step 3: Convert JSON to CSV
        print("\n" + "="*60)
        print("STEP 2: DATA PROCESSING")
        print("="*60)
        converter = JSONToCSVConverter(self.site, self.sport)
        csv_path = converter.convert(json_path)
        logger.info(f"Created CSV: {csv_path}")
        
        # Step 4: Process newsletters (optional)
        print("\n" + "="*60)
        print("STEP 3: NEWSLETTER INSIGHTS (Optional)")
        print("="*60)
        newsletter_path = self.data_dir / "raw" / f"newsletter_{self.slate_id}.txt"
        
        insights = None
        if click.confirm("Do you have newsletter insights to upload?"):
            print(f"\nPaste newsletter content into: {newsletter_path}")
            input("Press Enter when ready...")
            
            if newsletter_path.exists():
                parser = NewsletterParser()
                insights = parser.parse(newsletter_path)
                insights_path = self.data_dir / "processed" / f"insights_{self.slate_id}.json"
                with open(insights_path, 'w') as f:
                    json.dump(insights, f, indent=2)
                logger.info(f"Parsed insights saved to: {insights_path}")
        
        # Step 5: Modify projections if insights available
        base_projections_path = csv_path
        modified_projections_path = None
        
        if insights:
            print("\n" + "="*60)
            print("STEP 4: APPLYING INSIGHTS")
            print("="*60)
            modifier = ProjectionModifier()
            modified_projections_path = modifier.apply_insights(csv_path, insights)
            logger.info(f"Modified projections saved to: {modified_projections_path}")
        
        # Step 6: Build correlation matrix
        print("\n" + "="*60)
        print("STEP 5: BUILDING CORRELATIONS")
        print("="*60)
        correlation_builder = CorrelationBuilder(self.sport)
        correlation_matrix = correlation_builder.build(base_projections_path)
        
        # Step 7: Run simulations
        print("\n" + "="*60)
        print("STEP 6: RUNNING SIMULATIONS")
        print("="*60)
        simulator = MonteCarloSimulator(n_simulations=25000)
        
        # Run base simulations
        print("\nRunning base projections simulation...")
        base_results = simulator.run(
            base_projections_path,
            correlation_matrix,
            label="base"
        )
        
        # Run modified simulations if available
        modified_results = None
        if modified_projections_path:
            print("Running modified projections simulation...")
            modified_results = simulator.run(
                modified_projections_path,
                correlation_matrix,
                label="modified"
            )
        
        # Step 8: Optimize lineups
        print("\n" + "="*60)
        print("STEP 7: OPTIMIZING LINEUPS")
        print("="*60)
        optimizer = LineupOptimizer(self.site, self.sport)
        
        base_lineups = optimizer.optimize(base_results, n_lineups=20)
        modified_lineups = None
        if modified_results:
            modified_lineups = optimizer.optimize(modified_results, n_lineups=20)
        
        # Step 9: Select best lineup
        print("\n" + "="*60)
        print("STEP 8: SELECTING BEST LINEUP")
        print("="*60)
        selector = LineupSelector()
        
        if modified_lineups:
            best_lineup = selector.compare_and_select(base_lineups, modified_lineups)
            print("\n📊 Comparison Results:")
            print(f"Base lineup score: {base_lineups[0]['score']:.2f}")
            print(f"Modified lineup score: {modified_lineups[0]['score']:.2f}")
        else:
            best_lineup = base_lineups[0]
        
        # Save final lineup
        output_path = self.output_dir / f"lineup_{self.site}_{self.sport}_{self.slate_id}.json"
        with open(output_path, 'w') as f:
            json.dump(best_lineup, f, indent=2)
        
        print(f"\n✅ Best lineup saved to: {output_path}")
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        
        return best_lineup


@click.command()
@click.option('--site', prompt='DFS Site (draftkings/fanduel)', help='DFS site')
@click.option('--sport', prompt='Sport (nfl/nba/mlb/nhl)', help='Sport')
@click.option('--slate-id', help='Slate identifier (defaults to today)')
def main(site: str, sport: str, slate_id: Optional[str] = None):
    """Run the DFS optimization pipeline"""
    try:
        pipeline = DFSPipeline(site, sport, slate_id)
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
