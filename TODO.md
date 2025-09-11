# DFS Simulator - Todo List

## ✅ Completed
- [x] **JSON link creator and data file creator**
  - Generate Linestar API URLs for sport/site/period
  - Create empty JSON files for manual data paste

## 🔄 To Do

- [ ] **JSON extractor / CSV creator per sport / site**
  - Parse Linestar JSON into Player objects
  - Extract projections, salaries, ownership data
  - Sport-specific field mapping (MMA, NASCAR, NFL)
  - Export clean CSV with standardized columns

- [ ] **Newsletter processor; JSON created from LLM** 
  - Parse newsletter text for player insights
  - Generate structured JSON with targets/fades/locks
  - LLM integration for natural language processing
  - Standardized modifier format across sources

- [ ] **CSV modifier based on newsletter JSON**
  - Apply newsletter adjustments to projections
  - Boost targets (1.1-1.3x), fade concerns (0.7-0.9x)
  - Modify ownership projections for leverage
  - Export modified CSV for simulation input

- [ ] **Correlation matrix**
  - QB-WR, QB-TE positive correlations (same team)
  - Game stacks and opposing player relationships
  - Sport-specific correlation rules
  - Matrix format for simulation engine

- [ ] **Simulator based on CSV modifications & correlations**
  - Monte Carlo simulation (25,000+ runs)
  - Variance modeling with player correlations
  - Track 95th-99th percentile outcomes for GPP
  - Support for modified projections + newsletter insights

- [ ] **Simulator based on basic projections**
  - Fallback simulation without correlations
  - Pure variance modeling from projections
  - Independent player simulation
  - Baseline for comparison testing

- [ ] **Lineup optimizer based on simulations**
  - Target tail outcomes (not mean optimization)
  - Multi-lineup generation with diversity
  - Site-specific roster constraints (DraftKings)
  - Ownership-based differentiation strategy