#!/usr/bin/env python3
import sys
sys.path.append('../..')  # Add parent directory to path
"""
🎯 DFS LINEUP OPTIMIZER
Run Monte Carlo simulations and generate optimal lineups.

Usage:
    python optimize.py --sport mma --pid 466
    python optimize.py --sport nfl --pid week1 --entries 50
    python optimize.py --sport nba --pid 2024-12-15 --simulations 50000

This script:
    ✅ Loads processed CSV data
    ✅ Runs Monte Carlo simulations (25,000 default)
    ✅ Generates field simulation for uniqueness
    ✅ Optimizes lineups for GPP success
    ✅ Exports lineups and analysis
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time
from src.models.site import SiteCode

from src.optimization.base_optimizer import BaseOptimizer
from src.simulation.simulator import Simulator


def create_optimizer(sport: str, players: List = None) -> BaseOptimizer:
    """Factory function to create sport-specific optimizer."""
    optimizers = {
        'mma': lambda: _import_mma_optimizer()(players or [], sport),
        # 'nfl': lambda: _import_nfl_optimizer()(players or [], sport),  # Future implementation
        # 'nba': lambda: _import_nba_optimizer()(players or [], sport),  # Future implementation
    }
    
    optimizer_factory = optimizers.get(sport.lower())
    if not optimizer_factory:
        raise ValueError(f"Sport '{sport}' not yet supported")
    
    return optimizer_factory()


def _import_mma_optimizer():
    """Import MMA optimizer to avoid circular imports."""
    from src.optimization.mma_optimizer import MMAOptimizer
    return MMAOptimizer


def load_processed_data(sport: str, pid: str, site: SiteCode = SiteCode.DK) -> pd.DataFrame:
    """Load processed CSV data."""
    data_path = Path(f"data/{sport}/{pid}/{site.value}/csv")
    csv_file = data_path / "extracted.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(
            f"No processed data found: {csv_file}\n"
            f"Run: python process_data.py --sport {sport} --pid {pid} --site {site.value}"
        )
    
    df = pd.read_csv(csv_file)
    print(f"📄 Loaded {len(df)} players from: {csv_file}")
    return df


def display_lineup_summary(lineups: List, sport: str):
    """Display lineup generation summary."""
    if not lineups:
        print("❌ No lineups generated")
        return
    
    print(f"\n🎯 {sport.upper()} LINEUP SUMMARY")
    print("=" * 60)
    print(f"Generated: {len(lineups)} lineups")
    
    # Overall stats
    total_salaries = [lineup.total_salary for lineup in lineups]
    total_ownerships = [lineup.total_ownership for lineup in lineups]
    gpp_scores = [lineup.gpp_score for lineup in lineups if hasattr(lineup, 'gpp_score')]
    
    print(f"\n💰 Salary Usage:")
    print(f"   Range: ${min(total_salaries):,} - ${max(total_salaries):,}")
    print(f"   Average: ${np.mean(total_salaries):,.0f}")
    
    print(f"\n👥 Ownership:")
    print(f"   Range: {min(total_ownerships):.1f}% - {max(total_ownerships):.1f}%")
    print(f"   Average: {np.mean(total_ownerships):.1f}%")
    
    if gpp_scores:
        print(f"\n📊 GPP Scores:")
        print(f"   Range: {min(gpp_scores):.1f} - {max(gpp_scores):.1f}")
        print(f"   Average: {np.mean(gpp_scores):.1f}")
    
    # Top lineup details
    if hasattr(lineups[0], 'gpp_score'):
        best_lineup = max(lineups, key=lambda x: x.gpp_score)
        print(f"\n🏆 BEST LINEUP (GPP Score: {best_lineup.gpp_score:.1f})")
        print(f"   Salary: ${best_lineup.total_salary:,}")
        print(f"   Ownership: {best_lineup.total_ownership:.1f}%")
        if hasattr(best_lineup, 'percentile_95'):
            print(f"   95th Percentile: {best_lineup.percentile_95:.1f}")
        if hasattr(best_lineup, 'percentile_99'):
            print(f"   99th Percentile: {best_lineup.percentile_99:.1f}")


def export_lineups(lineups: List, sport: str, pid: str, site: SiteCode, output_format: str = 'csv', contest_type: str = 'gpp') -> Path:
    """Export lineups to file."""
    output_dir = Path(f"data/{sport}/{pid}/{site.value}/lineups")
    output_dir.mkdir(exist_ok=True)
    
    if output_format.lower() == 'csv':
        return _export_lineups_csv(lineups, output_dir, pid, contest_type)
    elif output_format.lower() == 'dk':
        return _export_lineups_dk(lineups, output_dir, pid, contest_type)
    else:
        raise ValueError(f"Unsupported export format: {output_format}")


def _export_lineups_csv(lineups: List, output_dir: Path, pid: str, contest_type: str = 'gpp') -> Path:
    """Export lineups to detailed CSV format."""
    output_file = output_dir / f"lineups_{pid}_{contest_type}.csv"
    
    # Sort lineups by GPP score (highest first)
    if lineups and hasattr(lineups[0], 'gpp_score'):
        lineups = sorted(lineups, key=lambda x: x.gpp_score, reverse=True)
    
    lineup_data = []
    for i, lineup in enumerate(lineups, 1):
        for j, player in enumerate(lineup.players, 1):
            row = {
                'lineup_id': i,
                'position': j,
                'player_name': player.name,
                'salary': player.salary,
                'projection': getattr(player, 'adjusted_projection', player.projection),
                'ownership': getattr(player, 'adjusted_ownership', player.ownership),
                'player_id': player.player_id,
            }
            
            # Add lineup-level metrics to each row
            if hasattr(lineup, 'gpp_score'):
                row['gpp_score'] = lineup.gpp_score
            if hasattr(lineup, 'percentile_25'):
                row['percentile_25'] = lineup.percentile_25
            if hasattr(lineup, 'percentile_50'):
                row['percentile_50'] = lineup.percentile_50
            if hasattr(lineup, 'percentile_75'):
                row['percentile_75'] = lineup.percentile_75
            if hasattr(lineup, 'percentile_95'):
                row['percentile_95'] = lineup.percentile_95
            if hasattr(lineup, 'percentile_99'):
                row['percentile_99'] = lineup.percentile_99
            if hasattr(lineup, 'leverage_score'):
                row['leverage_score'] = lineup.leverage_score
            if hasattr(lineup, 'uniqueness_score'):
                row['uniqueness_score'] = lineup.uniqueness_score
            
            lineup_data.append(row)
    
    df = pd.DataFrame(lineup_data)
    df.to_csv(output_file, index=False)
    print(f"💾 Exported detailed lineups: {output_file}")
    return output_file


def _export_lineups_dk(lineups: List, output_dir: Path, pid: str, contest_type: str = 'gpp') -> Path:
    """Export lineups in DraftKings upload format."""
    output_file = output_dir / f"dk_upload_{pid}_{contest_type}.csv"
    
    # This would need sport-specific position mapping
    # For now, just basic format
    lineup_data = []
    for i, lineup in enumerate(lineups, 1):
        row = {'Entry ID': i}
        for j, player in enumerate(lineup.players):
            row[f'Player {j+1}'] = f"{player.name} ({player.player_id})"
        lineup_data.append(row)
    
    df = pd.DataFrame(lineup_data)
    df.to_csv(output_file, index=False)
    print(f"💾 Exported DK upload format: {output_file}")
    return output_file


def main():
    """Main optimization workflow."""
    parser = argparse.ArgumentParser(
        description="🎯 DFS Lineup Optimizer - Generate optimal lineups via simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--sport', type=str, required=True,
                       choices=['mma', 'nfl', 'nba'],
                       help='Sport to optimize lineups for')
    parser.add_argument('--pid', type=str, required=True,
                       help='Contest/event identifier')
    parser.add_argument('--site', type=str, default='dk',
                       choices=['dk', 'fd'],
                       help='DFS site: dk=DraftKings, fd=FanDuel (default: dk)')
    parser.add_argument('--entries', type=int, default=20,
                       help='Number of lineups to generate (default: 20)')
    parser.add_argument('--simulations', type=int, default=25000,
                       help='Number of Monte Carlo simulations (default: 25,000)')
    parser.add_argument('--cash-game', action='store_true',
                       help='Optimize for cash games (50/50s) instead of GPPs')
    parser.add_argument('--export-format', type=str, default='csv',
                       choices=['csv', 'dk'],
                       help='Export format (default: csv)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only display summary of existing lineups')
    
    args = parser.parse_args()
    
    print("🎯 DFS LINEUP OPTIMIZER")
    print("=" * 60)
    print(f"Sport: {args.sport.upper()}")
    print(f"PID: {args.pid}")
    print(f"Site: {args.site.upper()}")
    print(f"Entries: {args.entries}")
    print(f"Simulations: {args.simulations:,}")
    print(f"Contest Type: {'Cash Game' if args.cash_game else 'GPP Tournament'}")
    print("=" * 60)
    
    try:
        # Convert site string to enum
        site = SiteCode(args.site)
        
        # Load processed data
        print("\n1️⃣ Loading processed data...")
        df = load_processed_data(args.sport, args.pid, site)
        
        if args.summary_only:
            # Just display summary of existing lineups
            output_dir = Path(f"data/{args.sport}/{args.pid}/{site.value}/lineups")
            lineup_file = output_dir / f"lineups_{args.pid}.csv"
            if not lineup_file.exists():
                print(f"❌ No existing lineups found: {lineup_file}")
                return 1
            
            lineup_df = pd.read_csv(lineup_file)
            num_lineups = lineup_df['lineup_id'].nunique()
            print(f"📊 Found {num_lineups} existing lineups")
            return 0
        
        # Create optimizer
        print(f"\n2️⃣ Initializing {args.sport.upper()} optimizer...")
        optimizer = create_optimizer(args.sport)
        optimizer.load_players_from_dataframe(df)
        optimizer.cash_game_mode = args.cash_game
        
        print(f"   ✅ Loaded {len(optimizer.players)} players")
        
        # Run Monte Carlo simulations
        print(f"\n3️⃣ Running Monte Carlo simulations...")
        start_time = time.time()
        
        simulator = Simulator(n_simulations=args.simulations)
        
        # Simulate each player
        print(f"   🎲 Simulating {len(optimizer.players)} players × {args.simulations:,} runs...")
        for player in optimizer.players:
            scores = simulator.simulate_player(player, use_cache=False)
            player.simulated_scores = scores
        
        sim_time = time.time() - start_time
        total_sims = len(optimizer.players) * args.simulations
        print(f"   ✅ Completed {total_sims:,} simulations in {sim_time:.1f}s")
        
        # Generate field for uniqueness scoring
        print(f"\n4️⃣ Generating field simulation...")
        field_start = time.time()
        optimizer.generate_field()  # Generate opponent lineups
        field_time = time.time() - field_start
        print(f"   ✅ Generated field in {field_time:.1f}s")
        
        # Generate optimized lineups
        print(f"\n5️⃣ Optimizing lineups...")
        opt_start = time.time()
        lineups = optimizer.optimize_lineups(args.entries)
        opt_time = time.time() - opt_start
        
        print(f"   ✅ Generated {len(lineups)} lineups in {opt_time:.1f}s")
        
        # Export lineups
        print(f"\n6️⃣ Exporting lineups...")
        contest_type = "cash" if args.cash_game else "gpp"
        output_file = export_lineups(lineups, args.sport, args.pid, site, args.export_format, contest_type)
        
        # Display summary
        display_lineup_summary(lineups, args.sport)
        
        total_time = time.time() - start_time
        print(f"\n🎯 OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total Time: {total_time:.1f}s")
        print(f"Output File: {output_file}")
        print(f"\n💡 Next Steps:")
        print("1. Review lineup analysis above")
        print("2. Check detailed lineup file for player selections")
        print("3. Upload to DraftKings when ready")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())