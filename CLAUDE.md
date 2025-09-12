# Claude AI Assistant Context

This document provides context for AI assistants (particularly Claude) working on the DFS Simulator project.

## Project Overview

This is a Daily Fantasy Sports (DFS) lineup optimizer designed for tournament play (GPPs - Guaranteed Prize Pools). The system uses advanced statistical methods to generate high-upside lineups that balance expected value with uniqueness.

## Core Architecture

### Data Flow
1. **JSON Input** → Manual data collection from various sources
2. **CSV Processing** → Standardized format for projections, salaries, ownership
3. **Newsletter Adjustments** → Expert insights modify base projections
4. **Monte Carlo Engine** → 25,000+ simulations with variance modeling
5. **Field Generation** → Model opponent lineups for uniqueness scoring
6. **Optimization** → Select lineups optimizing for tail outcomes (95th-99th percentile)

### Key Technical Components

#### Simulation Engine
- Uses numpy/pandas for vectorized operations
- Implements player variance through normal/lognormal distributions
- Correlation matrix for player dependencies (stacks, game correlations)
- Tracks cumulative distributions for percentile calculations

#### Correlation Framework
- **Positive correlations**: QB-WR, QB-TE, RB-DST (same team)
- **Negative correlations**: RB-DST (opposing), QB-opposing DST
- **Game stacks**: Multiple players from same game
- **Leverage plays**: Low-owned alternatives to chalk

##### MMA-Specific Correlations
- **Perfect negative correlation**: Direct opponents cannot both score (-0.8 to -1.0)
- **Strict constraint**: Never roster fighters from both sides of the same fight
- **ITD correlation**: Finishing potential affects lineup construction
- **Ownership anti-correlation**: Chalk vs contrarian fighter selection

#### Optimization Approach
- Not traditional linear programming (which optimizes for mean)
- Instead: simulation-based optimization for high-variance outcomes
- Scoring function: `tail_utility - ownership_penalty - duplication_penalty`
- Multi-lineup diversity through controlled overlap constraints

## Code Style Guidelines

### Python Standards
- Python 3.11+ with type hints
- Black formatter (100 char line length)
- Ruff linter with strict ruleset
- Docstrings for all public functions
- Pytest for testing with 80%+ coverage goal

### File Organization
- `src/` contains all production code
- Each module has single responsibility
- Interfaces defined with Protocol classes
- Configuration via YAML files, not hardcoded

### Performance Considerations
- Vectorize operations where possible
- Cache correlation matrices
- Use generators for large datasets
- Profile before optimizing

## Common Tasks

### Adding a New Sport
1. Create sport-specific configuration in `config/sports/`
2. Define position requirements and salary cap
3. Implement correlation rules in `src/simulation/correlations.py`
4. Add sport-specific variance models

### Implementing a New Optimizer
1. Inherit from `BaseOptimizer` class
2. Implement `score_lineup()` method
3. Define constraint validation
4. Add to optimizer factory

### Processing Newsletter Insights
1. Parse JSON structure for targets/fades/volatile
2. Apply projection multipliers (typically 1.1-1.3x for targets)
3. Adjust variance for volatile players
4. Modify ownership projections for fades

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external data sources
- Validate statistical properties of simulations

### Integration Tests
- End-to-end lineup generation
- Verify constraint satisfaction
- Check output formats

### Backtesting
- Historical data validation
- Compare actual vs predicted outcomes
- Track ROI and win rates

## Common Pitfalls to Avoid

1. **Over-optimization**: Don't chase perfect lineups, embrace variance
2. **Ignoring correlations**: Independent player modeling fails in DFS
3. **Mean-focused optimization**: Tournaments require tail-outcome focus
4. **Excessive chalk**: Balance optimal plays with differentiation
5. **Single-lineup thinking**: Build portfolios, not individual lineups

## Useful Commands

```bash
# Install dependencies
uv pip install -e .

# Run tests
pytest

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Generate lineups
python -m dfs-sim.cli generate --week 1 --sport nfl --entries 20

# Run backtesting
python -m dfs-sim.cli backtest --start 2023-09-01 --end 2024-01-01
```

## External Resources

- DFS correlation research papers
- Ownership projection methodologies
- Tournament game theory concepts
- Sports-specific variance studies

### Sport Configuration Reference

See `/archive/rules/sport_rules.py` for comprehensive sport-specific rules including:

#### MMA Configuration
- **Roster**: 6 fighters, $50K salary cap
- **Core constraint**: Never roster opponents from same fight
- **Composition strategy**: 2 favorites (≤-250 ML) + 2 mid-tier + 1-2 live dogs (+120 to +300)
- **Ownership targets**: ≥2 ≤20% owned, ≥1 ≤10% owned, cap total ≤140%
- **Ceiling formula**: `adj_ceil = CEIL * (1 - 0.20 * ownership) * (1 + 0.4 * ITD_prob)`
- **Key factors**: ITD (Inside The Distance) finishing potential, grappling control, volatility

#### Other Sports
- **NFL**: Stacking rules (QB + pass-catchers), correlation constraints
- **NBA**: Minutes ceiling focus, negative correlation avoidance  
- **MLB**: Hitter stacking (4-4, 5-3), pitcher vs stack rules
- **NHL**: Line stacks, PP1 correlation, goalie constraints
- **NASCAR**: Dominator strategy, position/PD balance, team limits
- **PGA**: Ceiling-first approach, wave correlation, ownership leverage

## Questions to Ask When Implementing Features

1. How does this affect lineup diversity?
2. What's the computational cost at 25k+ simulations?
3. Does this properly account for correlation?
4. How does this perform in backtesting?
5. Is the configuration flexible enough for different contest types?

## Next Development Priorities

1. Complete core simulation engine
2. Implement correlation matrix builder
3. Create field generation module
4. Build lineup optimization algorithm
5. Add backtesting framework
6. Create CLI interface
7. Develop web dashboard (future)