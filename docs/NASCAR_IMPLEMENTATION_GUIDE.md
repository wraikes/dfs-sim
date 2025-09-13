# NASCAR DFS Implementation Guide

## Overview
This document provides step-by-step instructions to implement NASCAR DFS optimization following the proven MMA pipeline. We successfully built MMA with Vegas odds integration, cash game modes, and Monte Carlo simulation - now replicate this exact process for NASCAR.

## Background Context
We just completed the MMA implementation with these key features:
- **Vegas odds integration**: Floor/ceiling adjustments based on win probabilities
- **Cash game vs GPP modes**: Different optimization targets (consistency vs ceiling)
- **Monte Carlo simulation**: 5000+ simulations with correlation modeling
- **Complete pipeline**: setup_data → process_data → optimize → export lineups

## NASCAR Implementation Steps

### 1. Update Setup Data (src/cli/setup_data.py)

**Current Status**: Only supports MMA
**Required Changes**:

```python
# Add NASCAR sport configuration
elif sport.lower() == 'nascar':
    return NASCARDataSetup(pid, site)

class NASCARDataSetup(BaseDataSetup):
    def get_contest_urls(self) -> Dict[str, str]:
        """NASCAR-specific data source URLs."""
        return {
            'linestar': f'https://linestarapp.com/nascar/{self.pid}',
            'dk_contest': f'https://www.draftkings.com/contest/nascar/{self.pid}',
            'fd_contest': f'https://www.fanduel.com/contests/nascar/{self.pid}',
            'newsletter': 'Manual - paste NASCAR newsletter content'
        }
    
    def get_empty_files(self) -> Dict[str, str]:
        return {
            f'{self.site}_{self.pid}.json': 'LineStar NASCAR contest JSON data',
            'newsletter_signals.json': 'NASCAR newsletter insights'
        }
```

**Key NASCAR Considerations**:
- **Data Sources**: LineStar, DraftKings contests, RotoGrinders newsletters
- **Track Types**: Superspeedway vs intermediate vs road courses (different strategies)
- **File Structure**: `data/nascar/{pid}/{site}/json/`

---

### 2. Update Process Data (src/cli/process_data.py)

**Current Status**: Only extracts MMA data from SalaryContainerJson
**Required Changes**:

#### A. Add NASCAR JSON Extraction
```python
def extract_nascar_data(self, json_file: Path) -> pd.DataFrame:
    """Extract NASCAR driver data from LineStar JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Parse SalaryContainerJson for NASCAR drivers
    salary_data = json.loads(data['SalaryContainerJson'])
    drivers = []
    
    for player_data in salary_data:
        driver = {
            'player_id': player_data.get('ID', 0),
            'name': player_data.get('Name', ''),
            'salary': player_data.get('Salary', 0),
            'projection': player_data.get('FPPG', 0),
            'ownership': player_data.get('Own%', 0),
            'position': player_data.get('Pos', ''),  # Should be 'D' for Driver
            
            # NASCAR-specific fields
            'team': player_data.get('Team', ''),
            'manufacturer': player_data.get('Manufacturer', ''),  # Chevy/Ford/Toyota
            'starting_position': player_data.get('StartPos', 0),
            'qualifying_speed': player_data.get('QualSpeed', 0),
            'practice_speeds': player_data.get('PracticeSpeeds', []),
            'recent_finishes': player_data.get('RecentFinishes', []),
            'track_history': player_data.get('TrackHistory', {}),
            
            # Vegas odds (if available)
            'win_odds': player_data.get('WinOdds', 0),
            'top5_odds': player_data.get('Top5Odds', 0),
            'top10_odds': player_data.get('Top10Odds', 0),
        }
        drivers.append(driver)
    
    return pd.DataFrame(drivers)
```

#### B. Add NASCAR Newsletter Processing
```python
def process_nascar_newsletter(self, df: pd.DataFrame) -> pd.DataFrame:
    """Apply NASCAR newsletter insights."""
    # Process targets/fades for specific track types
    # Example signals: "Dominator plays", "Value picks", "Superspeedway stackers"
    
    for signal in self.newsletter_data.get('targets', []):
        name = signal['name']
        confidence = signal['confidence']
        reason = signal['reason']
        
        # Apply NASCAR-specific multipliers
        if 'dominator' in reason.lower():
            # Higher ceiling for dominator candidates
            multiplier = 1.2 + (confidence * 0.15)
        elif 'value' in reason.lower():
            # Floor boost for value plays
            multiplier = 1.1 + (confidence * 0.1)
        else:
            multiplier = 1.1 + (confidence * 0.1)
        
        # Apply to matching driver
        mask = df['name'].str.contains(name, case=False, na=False)
        if mask.any():
            df.loc[mask, 'updated_projection'] *= multiplier
```

#### C. Add NASCAR-Specific Derived Metrics
```python
def calculate_nascar_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate NASCAR-specific performance metrics."""
    
    # Position differential potential (key NASCAR metric)
    df['position_differential'] = df['starting_position'] - 15  # Avg finish target
    df['pd_upside'] = np.maximum(0, df['position_differential']) * 0.5  # Bonus for advancing
    
    # Track type adjustments
    track_type = self.get_track_type()  # Superspeedway/Intermediate/Road
    
    if track_type == 'superspeedway':
        # Higher variance, more randomness
        df['variance_multiplier'] = 1.4
        df['ceiling_boost'] = 0.15  # 15% ceiling boost
    elif track_type == 'road':
        # Road course specialists
        df['road_course_bonus'] = df['track_history'].apply(self.extract_road_performance)
    else:
        # Intermediate tracks - standard
        df['variance_multiplier'] = 1.0
    
    # Manufacturer correlation setup
    df['manufacturer_group'] = df['manufacturer'].map({'Chevrolet': 'CHEVY', 'Ford': 'FORD', 'Toyota': 'TOYOTA'})
    
    return df
```

---

### 3. Create NASCAR Optimizer (src/optimization/nascar_optimizer.py)

**Pattern**: Copy `mma_optimizer.py` structure exactly
**Required Changes**:

```python
class NASCAROptimizer(BaseOptimizer):
    """NASCAR-specific DFS optimizer."""
    
    def __init__(self, players: List[Player], sport: str = 'nascar', field_size: int = 10000):
        super().__init__(players, sport, field_size)
    
    def _get_sport_constraints(self) -> SportConstraints:
        """NASCAR-specific constraints."""
        return SportConstraints(
            salary_cap=50000,
            roster_size=6,  # 6 drivers
            max_salary_remaining=1000,
            min_salary_remaining=200,
            max_lineup_ownership=180.0,  # Higher than MMA (more chalk acceptable)
            min_leverage_plays=1,        # At least 1 low-owned driver
            max_lineup_overlap=0.5,      # Allow more overlap than MMA
        )
    
    def validate_lineup_constraints(self, lineup: BaseLineup) -> bool:
        """NASCAR-specific lineup validation."""
        
        # 1. Must have exactly 6 drivers
        if len(lineup.players) != 6:
            return False
        
        # 2. Salary cap compliance
        if not (48000 <= lineup.total_salary <= 50000):
            return False
        
        # 3. Maximum 2 drivers from same manufacturer (diversification)
        manufacturer_counts = {}
        for player in lineup.players:
            mfg = getattr(player, 'manufacturer', 'UNKNOWN')
            manufacturer_counts[mfg] = manufacturer_counts.get(mfg, 0) + 1
            if manufacturer_counts[mfg] > 2:
                return False
        
        # 4. At least 1 driver starting outside top 15 (leverage play)
        starting_positions = [getattr(p, 'starting_position', 1) for p in lineup.players]
        if not any(pos > 15 for pos in starting_positions):
            return False
        
        return True
    
    def _generate_single_lineup(self) -> Optional[BaseLineup]:
        """Generate single NASCAR lineup using strategy tiers."""
        
        # Categorize drivers by strategy
        dominators = [p for p in self.players if getattr(p, 'starting_position', 99) <= 5]  # Top 5 start
        mid_tier = [p for p in self.players if 6 <= getattr(p, 'starting_position', 99) <= 15]
        value_plays = [p for p in self.players if getattr(p, 'starting_position', 99) > 15]
        
        # NASCAR lineup construction strategy
        # 2 dominators + 2 mid-tier + 2 value plays
        selected = []
        
        try:
            # Select 2 dominators (high-priced, top starting positions)
            selected.extend(random.sample(dominators, min(2, len(dominators))))
            
            # Select 2 mid-tier drivers
            selected.extend(random.sample(mid_tier, min(2, len(mid_tier))))
            
            # Select 2 value plays (leverage)
            selected.extend(random.sample(value_plays, min(2, len(value_plays))))
            
            # Fill to 6 drivers if needed
            while len(selected) < 6:
                remaining = [p for p in self.players if p not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
                else:
                    break
            
            if len(selected) == 6:
                lineup = NASCARLineup(players=selected)
                if self.validate_lineup_constraints(lineup):
                    return lineup
        
        except (ValueError, IndexError):
            pass
        
        return None
```

---

### 4. Create NASCAR Correlation Matrix (src/simulation/correlations.py)

**Required Addition**:

```python
class NASCARCorrelationBuilder(BaseCorrelationBuilder):
    """NASCAR correlation matrix with manufacturer and track-specific rules."""
    
    def build_matrix(self) -> np.ndarray:
        """Build NASCAR correlation matrix."""
        n = len(self.players)
        matrix = np.eye(n)
        
        # 1. Manufacturer correlations (teammates perform similarly)
        for i, driver1 in enumerate(self.players):
            mfg1 = getattr(driver1, 'manufacturer', '')
            
            for j, driver2 in enumerate(self.players):
                if i >= j:
                    continue
                    
                mfg2 = getattr(driver2, 'manufacturer', '')
                
                # Same manufacturer: positive correlation
                if mfg1 == mfg2 and mfg1 in ['Chevrolet', 'Ford', 'Toyota']:
                    matrix[i][j] = 0.15
                    matrix[j][i] = 0.15
        
        # 2. Starting position correlations
        for i, driver1 in enumerate(self.players):
            start1 = getattr(driver1, 'starting_position', 99)
            
            for j, driver2 in enumerate(self.players):
                if i >= j or abs(matrix[i][j]) > 0.1:
                    continue
                    
                start2 = getattr(driver2, 'starting_position', 99)
                
                # Similar starting positions: slight positive correlation
                if abs(start1 - start2) <= 3:
                    matrix[i][j] = 0.10
                    matrix[j][i] = 0.10
        
        # 3. Track type specific correlations
        track_type = self.get_track_type()
        
        if track_type == 'superspeedway':
            # Higher overall correlation (pack racing)
            matrix *= 1.2
            matrix = np.clip(matrix, -1, 1)
        elif track_type == 'road':
            # Road course specialists correlate
            for i, driver1 in enumerate(self.players):
                road_skill1 = getattr(driver1, 'road_course_rating', 0)
                
                for j, driver2 in enumerate(self.players):
                    if i >= j:
                        continue
                    
                    road_skill2 = getattr(driver2, 'road_course_rating', 0)
                    
                    if road_skill1 > 7 and road_skill2 > 7:  # Both road specialists
                        matrix[i][j] = 0.25
                        matrix[j][i] = 0.25
        
        return matrix

# Update factory function
def build_correlation_matrix(sport: str, players: List[Player]) -> Tuple[np.ndarray, List[CorrelationRule]]:
    """Build correlation matrix for any sport."""
    if sport.lower() == 'mma':
        builder = MMACorrelationBuilder(players)
    elif sport.lower() == 'nascar':
        builder = NASCARCorrelationBuilder(players)
    else:
        # Default: no correlations
        n = len(players)
        return np.eye(n), []
    
    matrix = builder.build_matrix()
    rules = builder.get_rules()
    return matrix, rules
```

---

### 5. Update Variance Model (src/simulation/variance_model.py)

**Required Addition**:

```python
# Add to VarianceModel.__post_init__()
if self.position_variance is None:
    self.position_variance = {
        # ... existing positions ...
        
        # NASCAR
        Position.DRIVER: 0.35,  # High variance sport (crashes, mechanical issues)
    }

# Add to position_distributions
if self.position_distributions is None:
    # ...existing logic...
    for pos in Position:
        if pos in [Position.DRIVER]:  # NASCAR uses lognormal for long tail events
            self.position_distributions[pos] = DistributionType.LOGNORMAL
        else:
            self.position_distributions[pos] = DistributionType.NORMAL

# Add NASCAR-specific variance adjustments
def simulate_score(self, player: Player, n_sims: int = 1) -> np.ndarray:
    # ... existing code ...
    
    # NASCAR-specific adjustments
    if player.position == Position.DRIVER:
        # Track type variance
        track_type = getattr(player, 'track_type', 'intermediate')
        
        if track_type == 'superspeedway':
            std_dev *= 1.3  # Higher variance (big wrecks possible)
        elif track_type == 'road':
            std_dev *= 0.9  # Lower variance (skill-based)
        
        # Starting position impact
        start_pos = getattr(player, 'starting_position', 20)
        if start_pos <= 5:
            std_dev *= 0.9  # Front runners more consistent
        elif start_pos >= 25:
            std_dev *= 1.2  # Back of pack more volatile
    
    # ... rest of existing simulation logic ...
```

---

### 6. Update CLI Integration

**Required Changes**:

#### A. Update optimize.py factory
```python
def create_optimizer(sport: str, players: List = None) -> BaseOptimizer:
    """Factory function to create sport-specific optimizer."""
    optimizers = {
        'mma': lambda: _import_mma_optimizer()(players or [], sport),
        'nascar': lambda: _import_nascar_optimizer()(players or [], sport),
    }
    
    optimizer_factory = optimizers.get(sport.lower())
    if not optimizer_factory:
        raise ValueError(f"Sport '{sport}' not yet supported")
    
    return optimizer_factory()

def _import_nascar_optimizer():
    """Import NASCAR optimizer to avoid circular imports."""
    from src.optimization.nascar_optimizer import NASCAROptimizer
    return NASCAROptimizer
```

#### B. Update process_data.py router
```python
def extract_data_for_sport(self, sport: str, json_file: Path) -> pd.DataFrame:
    """Route to sport-specific extraction."""
    if sport.lower() == 'mma':
        return self.extract_mma_data(json_file)
    elif sport.lower() == 'nascar':
        return self.extract_nascar_data(json_file)
    else:
        raise ValueError(f"Sport '{sport}' not supported")
```

---

### 7. Add NASCAR Position to Player Model

**Required Change** in `src/models/player.py`:

```python
class Position(Enum):
    # ... existing positions ...
    
    # NASCAR
    DRIVER = "DRIVER"
```

---

### 8. Create NASCAR Lineup Model

**Create** `src/models/nascar_lineup.py`:

```python
from .lineup import BaseLineup
from typing import List
from .player import Player

class NASCARLineup(BaseLineup):
    """NASCAR-specific lineup with 6 drivers."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # NASCAR-specific metrics
        self.num_dominators = sum(1 for p in self.players if getattr(p, 'starting_position', 99) <= 5)
        self.num_value_plays = sum(1 for p in self.players if getattr(p, 'starting_position', 99) > 20)
        self.manufacturer_diversity = len(set(getattr(p, 'manufacturer', 'UNK') for p in self.players))
        
        # Position differential upside
        total_pd_upside = sum(max(0, 15 - getattr(p, 'starting_position', 15)) for p in self.players)
        self.position_differential_upside = total_pd_upside
```

---

### 9. Testing Strategy

**Create** `tests/golden/test_nascar_optimize.py`:

```python
def test_nascar_optimize_golden():
    """Golden Test: NASCAR optimize.py generates lineups."""
    # Test with small simulation count
    result = subprocess.run([
        sys.executable, "src/cli/optimize.py",
        "--sport", "nascar",
        "--pid", "test_pid",
        "--site", "dk", 
        "--entries", "5",
        "--simulations", "1000"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    
    # Validate 6 drivers per lineup
    output_file = Path("data/nascar/test_pid/dk/lineups/lineups_test_pid_gpp.csv")
    df = pd.read_csv(output_file)
    
    # Each lineup should have 6 drivers
    lineup_ids = df['lineup_id'].unique()
    for lineup_id in lineup_ids:
        lineup_rows = df[df['lineup_id'] == lineup_id]
        assert len(lineup_rows) == 6, f"NASCAR lineup {lineup_id} should have 6 drivers"
```

---

### 10. Execution Checklist

When you're told "now do it", follow this exact sequence:

1. **Update setup_data.py** - Add NASCARDataSetup class
2. **Update process_data.py** - Add extract_nascar_data() method
3. **Create nascar_optimizer.py** - Copy MMA pattern with NASCAR constraints
4. **Update correlations.py** - Add NASCARCorrelationBuilder class
5. **Update variance_model.py** - Add DRIVER position and NASCAR variance rules
6. **Update optimize.py** - Add NASCAR to factory function
7. **Update player.py** - Add Position.DRIVER enum
8. **Create nascar_lineup.py** - NASCAR-specific lineup model
9. **Create test files** - Copy MMA golden tests for NASCAR
10. **Test end-to-end** - Run complete pipeline with sample data

---

### Key NASCAR-Specific Considerations

**Track Types**:
- **Superspeedway**: High variance, pack racing, manufacturer correlations strong
- **Intermediate**: Standard variance, qualifying position matters
- **Road Course**: Low variance, skill-based, specialists important

**Roster Construction**:
- 2 dominators (top 5 starting position)
- 2 mid-tier (positions 6-15) 
- 2 value plays (positions 16+)
- Max 2 drivers per manufacturer
- Salary cap: $50K

**Key Metrics**:
- Position Differential (starting position vs finish)
- Manufacturer correlation (teammates)
- Track type adjustments
- Starting position tiers

**Vegas Integration**:
- Win odds, Top 5 odds, Top 10 odds
- Use win probability for floor/ceiling like MMA
- Apply to position differential expectations

This follows the exact same pattern we used for MMA but adapted for NASCAR's unique characteristics. The end result should be a complete NASCAR DFS optimizer with both GPP and cash game modes.