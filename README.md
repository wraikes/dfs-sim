# DFS Simulator

A sophisticated Daily Fantasy Sports (DFS) lineup optimizer that uses Monte Carlo simulations, field modeling, and tail-outcome optimization to generate high-upside tournament lineups.

## Overview

This project implements a complete DFS optimization pipeline:

1. **Data Ingestion**: Pull baseline projections, salaries, and ownership from JSON sources
2. **Expert Adjustments**: Apply newsletter insights (target/fade/volatile) to tweak projections
3. **Monte Carlo Simulation**: Run 25,000+ simulations with player variances and correlations
4. **Field Modeling**: Generate opponent lineups to estimate chalk exposure and duplication risk
5. **Lineup Optimization**: Score lineups by 95th-99th percentile outcomes with uniqueness bonuses
6. **Selection**: Choose lineups that are high-ceiling, unique, and coherent

## Features

- **Variance Modeling**: Capture player outcome distributions and game-level correlations
- **Correlation Framework**: QB-WR stacks, game stacks, leverage plays, and wave effects
- **Ownership Leverage**: Balance EV with tournament leverage through ownership fades
- **Multi-Lineup Generation**: Create diverse lineup portfolios for multi-entry contests
- **Risk Management**: Track and limit exposure across lineups
- **Backtesting**: Validate strategies against historical results

## Project Structure

```
dfs-sim/
├── data/
│   ├── raw/              # Manual JSON inputs
│   ├── processed/        # Cleaned CSV files
│   └── cache/            # Correlation matrices, etc.
├── src/
│   ├── ingestion/        # Data loading and parsing
│   ├── adjustments/      # Newsletter insights processing
│   ├── simulation/       # Monte Carlo engine
│   ├── field/            # Opponent modeling
│   ├── optimization/     # Lineup generation
│   └── evaluation/       # Backtesting and analysis
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit and integration tests
└── outputs/              # Generated lineups and reports
```

## Installation

This project uses `uv` for fast Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/wraikes/dfs-sim.git
cd dfs-sim

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Quick Start

```python
# Example usage (coming soon)
from dfs_sim import DFSOptimizer

optimizer = DFSOptimizer(
    sport="nfl",
    contest_type="gpp",
    field_size=150000,
    max_entries=150
)

# Load data
optimizer.load_projections("data/raw/week1_projections.json")
optimizer.load_ownership("data/raw/week1_ownership.json")

# Apply adjustments
optimizer.apply_newsletter_insights("data/raw/week1_insights.json")

# Run simulations
optimizer.run_simulations(n_sims=25000)

# Generate lineups
lineups = optimizer.generate_lineups(
    n_lineups=20,
    optimize_for="p95",
    max_overlap=0.6
)
```

## Configuration

Key parameters can be configured in `config/settings.yaml`:

- Simulation parameters (number of iterations, variance models)
- Correlation settings (stack rules, game correlations)
- Optimization constraints (salary cap, position requirements)
- Output preferences (lineup formats, export options)

## Data Format

### Input JSON Structure

```json
{
  "players": [
    {
      "name": "Player Name",
      "position": "QB",
      "team": "BUF",
      "opponent": "MIA",
      "salary": 8000,
      "projection": 22.5,
      "ownership": 0.15,
      "variance": 0.25
    }
  ],
  "insights": {
    "targets": ["Player A", "Player B"],
    "fades": ["Player C"],
    "volatile": ["Player D"]
  }
}
```

## Development

```bash
# Run tests
pytest

# Run with example data
python -m dfs_sim.cli --week 1 --sport nfl

# Start Jupyter for analysis
jupyter notebook
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built for tournament DFS optimization
- Inspired by modern quantitative DFS strategies
- Designed for GPP (large-field tournament) success