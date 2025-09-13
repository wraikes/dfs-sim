# NFL DFS Pipeline Implementation Guide

This guide outlines how to implement the NFL DFS pipeline following the established architecture patterns from the NASCAR implementation.

## 🏈 NFL DFS Strategy Overview

### Core NFL Concepts
- **Stacking**: QB + pass-catchers from same team for correlated scoring
- **Bring-back stacks**: Include opposing player to hedge main stack
- **Game stacks**: Multiple players from same game (high total)
- **Game theory**: Leverage plays vs optimal chalk balance
- **Position variance**: QB (low) → RB → WR → TE → DST (high)

### NFL Roster Construction
- **Roster Size**: 9 players ($50K salary cap)
- **Positions**: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DST
- **Core Strategy**: Build around 1-2 game stacks, add leverage plays

## 📋 Implementation Checklist

### 1. Data Collection Setup (`setup_data.py`)

**Add NFL Configuration:**
```python
'nfl': {
    'name': 'NFL',
    'linestar_sport_id': 1,  # NFL LineStar ID
    'data_sources': [
        'LineStarContest',
        'NewsletterAnalysis',
        'VegasOdds',
        'WeatherData',
        'InjuryReports'
    ]
}
```

**Key Data Sources:**
- **DraftKings Contest JSON**: Salary/projections for all players
- **Vegas Odds**: Game totals, spreads, player props
- **Weather Data**: Wind, temperature, precipitation (affects passing)
- **Newsletter Signals**: Expert targets/fades with reasoning
- **Injury Reports**: Game-time decisions and snap count projections

### 2. Data Processing (`process_data.py`)

**Create NFLDataProcessor Class:**
```python
class NFLDataProcessor(BaseDataProcessor):
    """NFL-specific data processor with stacking analysis."""

    def load_raw_data(self) -> pd.DataFrame:
        # Parse DraftKings contest JSON for NFL players
        # Extract: player_id, name, position, team, opponent, salary, projection

    def calculate_sport_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # NFL-specific calculations:
        # 1. Game totals and pace analysis
        # 2. Weather impact modeling
        # 3. Target share and air yards calculations
        # 4. Snap count projections
        # 5. Red zone efficiency metrics
        # 6. Matchup advantages (vs position rankings)
```

**Key NFL Metrics to Calculate:**
- **Passing Game**: Air yards, target share, red zone targets
- **Running Game**: Carries, goal line work, snap count %
- **Game Environment**: Pace, total, weather impact
- **Matchup Data**: vs position rankings, pace differentials
- **Leverage Metrics**: Ownership vs projection value

### 3. NFL Correlation Model (`correlations.py`)

**Add NFLCorrelationBuilder:**
```python
class NFLCorrelationBuilder(BaseCorrelationBuilder):
    def build_correlation_matrix(self, players: List[Player]) -> np.ndarray:
        # 1. QB-Pass Catcher Correlations (same team)
        #    QB-WR1: +0.65, QB-WR2: +0.45, QB-TE: +0.35

        # 2. Game Stack Correlations
        #    Same game total: +0.25 to +0.40
        #    High total games: +0.35, low total: +0.15

        # 3. Negative Correlations
        #    RB vs opposing DST: -0.25
        #    QB vs opposing DST: -0.35

        # 4. Bring-back Stack Correlations
        #    QB1 + WR1 + opponent WR1: +0.20 between opponents

        # 5. Weather/Environment Correlations
        #    Bad weather: RB +0.15, passing game -0.20
```

**NFL-Specific Correlation Patterns:**
- **Primary Stack**: QB + WR/TE from same team (+0.65/+0.35)
- **Game Correlations**: Players in same high-total game (+0.35)
- **Bring-back**: Opposing WR/TE to hedge primary stack (+0.20)
- **Negative**: RB/QB vs opposing DST (-0.25/-0.35)
- **Weather**: Bad weather boosts RB/DST, hurts passing

### 4. NFL Variance Model (`variance_model.py`)

**Add NFL Position Variance:**
```python
# NFL position variance (higher = more volatile)
Position.QB: 0.18,      # Most consistent
Position.RB: 0.25,      # Moderate (usage dependent)
Position.WR: 0.32,      # High (target dependent)
Position.TE: 0.38,      # Very high (touchdown dependent)
Position.DST: 0.45,     # Highest (big play dependent)
```

**NFL Variance Adjustments:**
```python
elif player.position in [Position.QB, Position.RB, Position.WR, Position.TE, Position.DST]:
    # Weather adjustments
    wind_speed = player.metadata.get('wind_speed', 0)
    if wind_speed > 15:  # High wind
        if player.position == Position.QB:
            std_dev *= 1.25  # QB more volatile in wind
        elif player.position == Position.RB:
            std_dev *= 0.90  # RB benefits from wind (more carries)

    # Game total adjustments
    game_total = player.metadata.get('game_total', 47.0)
    if game_total >= 52.0:  # High total
        std_dev *= 0.85  # More predictable scoring
    elif game_total <= 42.0:  # Low total
        std_dev *= 1.15  # More unpredictable

    # Target share adjustments (WR/TE)
    if player.position in [Position.WR, Position.TE]:
        target_share = player.metadata.get('target_share', 0.15)
        if target_share >= 0.25:  # High target share
            std_dev *= 0.80  # More consistent
        elif target_share <= 0.10:  # Low target share
            std_dev *= 1.30  # Boom/bust
```

### 5. NFL Optimizer (`nfl_optimizer.py`)

**Create NFLOptimizer Class:**
```python
@dataclass
class NFLLineup(BaseLineup):
    # NFL-specific metrics
    primary_stack_players: int = 0      # QB + receivers from same team
    bring_back_players: int = 0         # Opposing players to primary stack
    game_stack_count: int = 0           # Players from same game
    total_game_exposure: float = 0.0    # Sum of game totals

class NFLOptimizer(BaseOptimizer):
    def _get_sport_constraints(self) -> SportConstraints:
        return SportConstraints(
            salary_cap=50000,
            roster_size=9,
            sport_rules={
                'positions': {
                    'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1
                },
                'stacking_rules': {
                    'min_primary_stack': 2,      # QB + at least 1 receiver
                    'max_primary_stack': 4,      # QB + up to 3 receivers
                    'max_same_game_players': 5,  # Max players from one game
                    'require_bring_back': False, # Optional bring-back player
                },
                'leverage_rules': {
                    'min_leverage_plays': 2,     # At least 2 <10% owned
                    'max_chalk_players': 4,      # Max 4 >25% owned
                    'max_total_ownership': 160,  # Total lineup ownership
                }
            }
        )
```

**NFL Lineup Generation Strategy:**
```python
def _generate_single_lineup(self) -> Optional[BaseLineup]:
    # 1. Select primary game stack (high total preferred)
    # 2. Build QB + 2-3 pass catchers from primary team
    # 3. Add bring-back player from opposing team (optional)
    # 4. Fill remaining positions with value/leverage plays
    # 5. Validate stacking rules and ownership constraints
```

### 6. Testing Implementation

**Create Golden Tests (`tests/golden/test_nfl_integration.py`):**
```python
def test_nfl_data_processing():
    # Test NFL JSON parsing from DraftKings format
    # Verify proper position mapping and salary extraction

def test_nfl_correlations():
    # Test QB-WR stacking correlations (+0.65)
    # Test game stack correlations (+0.35)
    # Test negative correlations (RB vs opposing DST: -0.25)

def test_nfl_optimization():
    # Test valid 9-player lineup generation
    # Test stacking rule compliance (QB + 2-3 receivers)
    # Test position requirements and salary constraints

def test_nfl_variance_model():
    # Test weather impact on variance
    # Test game total adjustments
    # Test target share impact on WR/TE variance
```

## 🎯 NFL-Specific Implementation Details

### Data Extraction Priorities

**Essential NFL Data Fields:**
- **Player**: position, team, opponent, salary, projection, ownership
- **Game Environment**: total, spread, weather (wind, temp, precipitation)
- **Usage Metrics**: snap %, target share, red zone targets, carries
- **Matchups**: vs position rank, pace differential, defensive efficiency
- **Props**: receiving/rushing/passing yards, touchdowns, completions

**Weather Impact Modeling:**
```python
# Wind affects passing accuracy
if wind_speed > 15 and position in [QB, WR, TE]:
    projection *= 0.85  # Reduce passing projections

# Cold affects ball handling
if temperature < 32 and position == RB:
    std_dev *= 1.10  # Increase fumble risk

# Precipitation boosts defense
if precipitation > 0.1 and position == DST:
    projection *= 1.15  # More turnovers expected
```

### Correlation Implementation

**QB-Stack Implementation:**
```python
def _build_qb_stack_correlations(self, players):
    for qb in [p for p in players if p.position == Position.QB]:
        qb_team = qb.team

        # Find pass catchers on same team
        receivers = [p for p in players
                    if p.team == qb_team and p.position in [WR, TE]]

        for receiver in receivers:
            correlation = 0.65 if receiver.position == Position.WR else 0.35
            # Apply correlation based on target share
            target_share = receiver.metadata.get('target_share', 0.15)
            correlation *= min(1.0, target_share / 0.20)  # Scale by usage
```

### Optimization Strategy

**NFL Lineup Construction Priority:**
1. **Select QB** from high-total game or chalk/leverage play
2. **Stack 2-3 pass catchers** from QB's team (WR1 + WR2/TE)
3. **Add bring-back** from opposing team if game total >50
4. **Fill RB positions** with volume plays and value
5. **Select DST** contrarian to popular teams
6. **Use FLEX** for additional leverage or stack completion

**Position-Specific Strategy:**
- **QB**: Target 25+ projected points, prefer high totals
- **RB**: Focus on volume (15+ carries) and goal line work
- **WR**: Target share >20% or leverage plays <8% owned
- **TE**: Red zone targets and plus matchups vs LBs
- **DST**: Home teams, vs backup QBs, weather games

## 🚀 Ready for Implementation

Following this guide will create a complete NFL DFS pipeline that:
- ✅ Extracts comprehensive NFL data from multiple sources
- ✅ Models proper QB-receiver stacking correlations
- ✅ Applies weather and game environment variance
- ✅ Generates optimal stacked lineups with leverage
- ✅ Validates all NFL-specific rules and constraints

The NFL implementation will be the most complex sport due to 9 positions and stacking requirements, but following the NASCAR pattern ensures a robust, scalable architecture.