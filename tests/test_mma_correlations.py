"""Test script for MMA correlation matrix."""

import sys
import os
sys.path.append('/home/wraikes/programming/portfolio/dfs-sim')

import numpy as np
import pandas as pd
from src.models import Player
from src.simulation.correlations import build_correlation_matrix, MMACorrelationBuilder


def load_mma_players() -> list:
    """Load MMA players from CSV data."""
    df = pd.read_csv('/home/wraikes/programming/portfolio/dfs-sim/data/mma/466/csv/dk_466_extracted.csv')
    
    players = []
    for _, row in df.iterrows():
        # Create Player object with metadata for MMA-specific fields
        metadata = {
            'ml_odds': row.get('ml_odds', 0),
            'itd_probability': row.get('itd_probability', 0.35),
            'itd_adjusted_ceiling': row.get('itd_adjusted_ceiling', row.get('updated_ceiling', row['ceiling']))
        }
        
        player = Player(
            player_id=int(row['player_id']),
            name=row['name'],
            position=row['position'],
            team=row['team'],
            salary=int(row['salary']),
            projection=float(row['updated_projection']),
            floor=float(row['updated_floor']),
            ceiling=float(row['updated_ceiling']),
            std_dev=float(row['std_dev']) if pd.notna(row['std_dev']) else 0.0,
            ownership=float(row['updated_ownership']),
            value=float(row['value']) if pd.notna(row['value']) else 0.0,
            opponent=row['opponent'],
            metadata=metadata
        )
        
        players.append(player)
    
    return players


def test_mma_correlations():
    """Test MMA correlation matrix generation."""
    print("🥊 Testing MMA Correlation Matrix")
    print("=" * 50)
    
    # Load MMA players
    players = load_mma_players()
    print(f"Loaded {len(players)} MMA fighters")
    
    # Build correlation matrix
    correlation_matrix, rules = build_correlation_matrix('mma', players)
    
    # Display summary
    print(f"\nCorrelation Summary:")
    print(f"  Total Players: {len(players)}")
    print(f"  Opponent Rules: {len([r for r in rules if r.rule_type == 'opponent'])}")
    print(f"  Total Rules: {len(rules)}")
    
    # Show matrix properties
    print(f"\nMatrix Properties:")
    print(f"  Shape: {correlation_matrix.shape}")
    print(f"  Min correlation: {correlation_matrix.min():.3f}")
    print(f"  Max correlation: {correlation_matrix.max():.3f}")
    print(f"  Non-zero correlations: {np.count_nonzero(correlation_matrix - np.eye(len(players)))}")
    
    # Show specific opponent correlations
    print(f"\nOpponent Correlations:")
    for rule in rules[:5]:  # Show first 5 rules
        player1 = next(p for p in players if p.player_id == rule.player1_id)
        player2 = next(p for p in players if p.player_id == rule.player2_id)
        print(f"  {player1.name} vs {player2.name}: {rule.correlation:.3f}")
    
    # Validate matrix properties
    print(f"\nValidation:")
    is_symmetric = np.allclose(correlation_matrix, correlation_matrix.T)
    diagonal_ones = np.allclose(np.diag(correlation_matrix), 1.0)
    valid_range = (correlation_matrix >= -1.0).all() and (correlation_matrix <= 1.0).all()
    
    print(f"  ✅ Symmetric: {is_symmetric}")
    print(f"  ✅ Diagonal = 1: {diagonal_ones}")
    print(f"  ✅ Valid range [-1,1]: {valid_range}")
    
    if is_symmetric and diagonal_ones and valid_range:
        print(f"\n🎉 Correlation matrix validation PASSED!")
    else:
        print(f"\n❌ Correlation matrix validation FAILED!")
    
    return correlation_matrix, rules, players


if __name__ == "__main__":
    matrix, rules, players = test_mma_correlations()