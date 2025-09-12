# Claude AI Assistant Context

This document provides context for AI assistants (particularly Claude) working on the DFS Simulator project.

## Project Overview

This is a Daily Fantasy Sports (DFS) lineup optimizer designed for tournament play (GPPs - Guaranteed Prize Pools). The system uses advanced statistical methods to generate high-upside lineups that balance expected value with uniqueness.

## Core Architecture

### Data Flow
1. **Manual Data Setup** → Create directories and empty files for data input
2. **Sport-Specific JSON Parsing** → Extract raw JSON data into standardized CSV format  
3. **LLM Newsletter Processing** → Parse newsletter insights into JSON signals, modify CSV in-place
4. **Monte Carlo Engine** → 25,000+ simulations with variance modeling
5. **Field Generation** → Model opponent lineups for uniqueness scoring
6. **Optimization** → Select lineups optimizing for tail outcomes (95th-99th percentile)

### CLI Pipeline Workflow

The complete data processing pipeline follows this exact sequence:

```bash
# Step 1: Setup data directories and provide URLs for manual data collection
python src/cli/setup_data.py --sport mma --pid 466 --site dk
# → Creates: data/mma/dk/466/raw/ with empty JSON files
# → Provides URLs for LineStar, newsletter data sources

# Step 2: Manual data input (copy/paste JSON data into empty files)
# → User pastes contest JSON into data/mma/dk/466/raw/dk_466.json  
# → User pastes newsletter into data/mma/dk/466/raw/newsletter_signals.json

# Step 3: Process raw data (JSON → CSV → Newsletter → Modified CSV)
python src/cli/process_data.py --sport mma --pid 466 --site dk
# → Extracts JSON salary/projection data into standardized CSV format
# → Uses LLM to parse newsletter text into structured JSON signals
# → Applies newsletter adjustments to modify CSV projections in-place
# → Outputs: data/mma/dk/466/csv/dk_466_extracted.csv

# Step 4: Generate optimized lineups
python src/cli/optimize.py --sport mma --pid 466 --site dk --entries 20
# → Loads processed CSV data
# → Runs Monte Carlo simulations with correlation matrix
# → Generates field of opponent lineups for uniqueness scoring  
# → Outputs optimal lineups: data/mma/dk/466/output/lineups_466.csv
```

#### Data Processing Components

**Sport-Specific JSON Extraction:**
- MMA: Parses LineStar JSON format (`SalaryContainerJson`, `MatchupData`)
- NFL: Will parse DraftKings/FanDuel contest JSON (future implementation)
- NBA: Will parse projection service JSON format (future implementation)

**LLM Newsletter Processing:**
- Parses unstructured newsletter text using AI language model
- Extracts targets, fades, volatile players with confidence scores
- Creates structured JSON: `{"targets": [...], "fades": [...], "volatile": [...]}`
- Applies projection multipliers: targets (+15-35%), fades (-15-30%), volatile (ceiling +10-25%)

**CSV Modification Pipeline:**
- Base CSV: player_id, name, salary, projection, floor, ceiling, ownership
- Newsletter signals applied: updated_projection, updated_ceiling, updated_ownership  
- Final output: Fully processed CSV ready for simulation/optimization

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

## Implementation Status

### ✅ COMPLETED - MMA System (Fully Operational)
**Monte Carlo + Optimization Pipeline Working**

**Core Components Built:**
- **Correlation Matrix**: MMA-specific with opponent anti-correlation (-0.85), ITD/ML odds patterns
- **Monte Carlo Simulator**: 10k+ simulations with Cholesky decomposition for correlated outcomes  
- **Variance Model**: Lognormal distribution for fighters (50% variance coefficient)
- **CSV Data Pipeline**: ITD probability calculation, ML odds extraction, adjusted ceiling formula
- **GPP Optimizer**: Sport rules compliance (2 fav + 2 mid + 2 dogs, no opponents, ownership leverage)

**Key Files:**
- `src/simulation/correlations.py` - MMA correlation matrix builder
- `src/simulation/simulator.py` - Monte Carlo engine with correlation support
- `src/optimization/mma_optimizer.py` - Sport rules optimizer  
- `run_mma_simulation.py` - Full pipeline demo (280k simulations → optimal lineups)

**Proven Results:**
- 28 fighters × 10k sims = 280,000 total calculations in 0.1s
- Generated lineups with 497+ 95th percentile scores
- 0% ownership leverage plays (massive GPP edge)
- Perfect sport rules compliance

### 🎯 NEXT PRIORITIES - Heavy Weight Sports

**NASCAR Implementation:**
- **Correlation patterns**: Manufacturer/team stacking, track position correlations
- **Dominator strategy**: P1-P12 vs P23+ (superspeedway vs intermediate tracks)
- **Complex constraints**: Max 2 drivers per team/OEM, position-based selection
- **Unique scoring**: Place differential (PD) + dominator points system

**NFL Implementation:**  
- **Stacking requirements**: QB + pass-catchers (2+ from same team), bring-back stacks
- **Game correlation**: Team totals, weather effects, pace of play
- **Position variance**: QB (low) → DST (high), leverage plays
- **Ownership complexity**: Chalk vs contrarian balance, lineup ownership caps

**Technical Challenges:**
- **NASCAR**: Multi-track correlation matrices (superspeedway vs intermediate vs road)
- **NFL**: Game stacking logic, weather/pace adjustments, showdown formats
- **Both**: More complex roster constraints than MMA (9 NFL positions vs 6 MMA)

## Next Development Priorities

1. **NASCAR Optimizer** - Superspeedway vs intermediate track logic ⚡
2. **NFL Optimizer** - Stacking engine + game correlation 🏈  
3. **CLI Interface** - `python -m dfs-sim.cli generate --sport nascar --entries 20`
4. **Multi-sport backtesting** - Historical ROI validation
5. **Contest integration** - DraftKings/FanDuel importing
6. **Field generation** - Opponent lineup modeling for uniqueness
7. **Web dashboard** - Visual lineup builder (future)