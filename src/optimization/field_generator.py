"""Field generation for opponent lineup modeling in GPPs."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..models.player import Player


@dataclass
class FieldLineup:
    """Represents an opponent's lineup in the field."""
    players: list[Player]  # Renamed from fighters to be sport-agnostic
    total_salary: int = 0
    total_ownership: float = 0
    lineup_id: int = 0

    def __post_init__(self):
        self.total_salary = sum(p.salary for p in self.players)
        self.total_ownership = sum(p.ownership for p in self.players)

    def get_player_ids(self) -> set[int]:
        """Return set of player IDs for overlap calculation."""
        return {p.player_id for p in self.players}


class BaseFieldGenerator(ABC):
    """Abstract base class for sport-specific field generators."""

    def __init__(self, players: list[Player]):
        self.players = players
        self._categorize_by_ownership()

    def _categorize_by_ownership(self):
        """Categorize players by ownership tiers."""
        self.chalk = [p for p in self.players if p.ownership >= 30]
        self.popular = [p for p in self.players if 20 <= p.ownership < 30]
        self.moderate = [p for p in self.players if 10 <= p.ownership < 20]
        self.contrarian = [p for p in self.players if 5 <= p.ownership < 10]
        self.leverage = [p for p in self.players if p.ownership < 5]

    @abstractmethod
    def _has_invalid_combinations(self, players: list[Player]) -> bool:
        """Check if lineup has sport-specific invalid combinations."""
        pass

    @abstractmethod
    def _select_by_ownership_probability(self, available: list[Player], n: int = 1) -> list[Player]:
        """Select players weighted by ownership probability."""
        pass

    @abstractmethod
    def generate_field(self, n_lineups: int = 10000) -> list[FieldLineup]:
        """Generate realistic field of opponent lineups."""
        pass


class MMAFieldGenerator(BaseFieldGenerator):
    """Generate realistic MMA opponent lineups based on ownership probabilities."""

    def __init__(self, players: list[Player]):
        super().__init__(players)
        self.salary_cap = 50000
        self.roster_size = 6

        # Pre-calculate opponent pairs to avoid
        self.opponent_pairs = self._build_opponent_pairs()

    def _build_opponent_pairs(self) -> set[tuple[int, int]]:
        """Build set of opponent pairs that can't be rostered together."""
        pairs = set()

        for p1 in self.players:
            if not p1.opponent:
                continue

            for p2 in self.players:
                if p1.player_id >= p2.player_id:
                    continue

                # Check if they are opponents
                if (p1.opponent.upper() == p2.name.upper() or
                    p1.opponent.upper() in p2.name.upper() or
                    p2.opponent and p2.opponent.upper() == p1.name.upper()):
                    pairs.add((p1.player_id, p2.player_id))
                    pairs.add((p2.player_id, p1.player_id))

        return pairs


    def _has_invalid_combinations(self, players: list[Player]) -> bool:
        """Check if lineup contains opponents (MMA-specific rule)."""
        player_ids = {p.player_id for p in players}

        for p1_id in player_ids:
            for p2_id in player_ids:
                if p1_id != p2_id and (p1_id, p2_id) in self.opponent_pairs:
                    return True
        return False

    def _select_by_ownership_probability(self, available: list[Player], n: int = 1) -> list[Player]:
        """Enhanced ownership probability selection with realistic field modeling."""
        if not available or n <= 0:
            return []

        # Get ownership values
        ownerships = np.array([p.ownership for p in available])

        # Enhanced ownership probability modeling
        # Real DFS players don't select exactly proportional to ownership - they have biases

        # Add base probability floor (even 0% owned players get selected sometimes)
        base_prob = 0.1
        ownerships = np.maximum(ownerships, base_prob)

        # Apply diminishing returns to high ownership (reduces chalk stacking)
        # Players > 50% owned don't scale linearly in selection
        ownership_adjusted = np.where(
            ownerships > 50,
            50 + (ownerships - 50) * 0.6,  # 60% scaling above 50%
            ownerships
        )

        # Add small random noise to simulate imperfect ownership prediction
        noise_factor = 0.15  # 15% noise
        noise = np.random.normal(1.0, noise_factor, len(ownership_adjusted))
        ownership_adjusted *= np.maximum(noise, 0.1)  # Keep positive

        # Convert to selection probabilities
        probabilities = ownership_adjusted / ownership_adjusted.sum()

        # Select without replacement
        n = min(n, len(available))
        selected_indices = np.random.choice(
            len(available),
            size=n,
            replace=False,
            p=probabilities
        )

        return [available[i] for i in selected_indices]

    def _generate_lineup_by_strategy(self, strategy: str) -> FieldLineup:
        """Generate a lineup based on ownership strategy.
        
        Args:
            strategy: 'chalk', 'balanced', or 'contrarian'
        """
        lineup = []
        used_ids = set()
        max_attempts = 100

        # Define strategy-specific ownership targets
        if strategy == 'chalk':
            # Chalk: 3-4 chalk/popular, 1-2 moderate/contrarian
            ownership_pools = [
                (self.chalk + self.popular, 0.7),  # 70% from high ownership
                (self.moderate + self.contrarian, 0.3)  # 30% from lower ownership
            ]
        elif strategy == 'balanced':
            # Balanced: Mix across all ownership levels
            ownership_pools = [
                (self.chalk + self.popular, 0.33),
                (self.moderate, 0.33),
                (self.contrarian + self.leverage, 0.34)
            ]
        elif strategy == 'contrarian':
            # Contrarian: Mostly low ownership with some mid-tier
            ownership_pools = [
                (self.leverage + self.contrarian, 0.67),  # 67% low ownership
                (self.moderate + self.popular, 0.33)  # 33% mid-tier
            ]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Build lineup based on ownership distribution
        attempts = 0
        while len(lineup) < 6 and attempts < max_attempts:
            attempts += 1

            # Select from appropriate pool based on strategy
            for pool, weight in ownership_pools:
                remaining = 6 - len(lineup)
                target_from_pool = int(remaining * weight) + (1 if remaining * weight % 1 > 0.5 else 0)
                target_from_pool = max(1, min(target_from_pool, remaining))

                candidates = [p for p in pool if p.player_id not in used_ids]
                if candidates:
                    selected = self._select_by_ownership_probability(candidates, 1)
                    if selected:
                        player = selected[0]
                        if not self._has_invalid_combinations(lineup + [player]):
                            lineup.append(player)
                            used_ids.add(player.player_id)
                            break

                if len(lineup) >= 6:
                    break

        # Fill remaining spots with any valid players
        while len(lineup) < 6:
            candidates = [p for p in self.players if p.player_id not in used_ids]
            if not candidates:
                break

            player = random.choice(candidates)
            if not self._has_invalid_combinations(lineup + [player]):
                lineup.append(player)
                used_ids.add(player.player_id)

        return FieldLineup(players=lineup[:6])

    def generate_chalk_lineup(self) -> FieldLineup:
        """Generate a typical chalk-heavy lineup that most opponents will play."""
        return self._generate_lineup_by_strategy('chalk')

    def generate_balanced_lineup(self) -> FieldLineup:
        """Generate a balanced lineup with mix of ownership levels."""
        return self._generate_lineup_by_strategy('balanced')

    def generate_contrarian_lineup(self) -> FieldLineup:
        """Generate a contrarian lineup with mostly low-owned plays."""
        return self._generate_lineup_by_strategy('contrarian')

    def generate_field(self, n_lineups: int = 10000) -> list[FieldLineup]:
        """Generate a realistic field of opponent lineups.
        
        Distribution:
        - 60% chalk-heavy lineups
        - 30% balanced lineups  
        - 10% contrarian lineups
        """
        field = []

        n_chalk = int(n_lineups * 0.60)
        n_balanced = int(n_lineups * 0.30)
        n_contrarian = n_lineups - n_chalk - n_balanced

        print(f"Generating field of {n_lineups:,} lineups...")
        print(f"  Chalk: {n_chalk:,} (60%)")
        print(f"  Balanced: {n_balanced:,} (30%)")
        print(f"  Contrarian: {n_contrarian:,} (10%)")

        # Generate chalk lineups
        for i in range(n_chalk):
            lineup = self.generate_chalk_lineup()
            lineup.lineup_id = i
            field.append(lineup)

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1:,} chalk lineups...")

        # Generate balanced lineups
        for i in range(n_balanced):
            lineup = self.generate_balanced_lineup()
            lineup.lineup_id = n_chalk + i
            field.append(lineup)

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1:,} balanced lineups...")

        # Generate contrarian lineups
        for i in range(n_contrarian):
            lineup = self.generate_contrarian_lineup()
            lineup.lineup_id = n_chalk + n_balanced + i
            field.append(lineup)

        print(f"✅ Generated {len(field):,} total lineups")

        # Calculate field statistics
        avg_ownership = np.mean([l.total_ownership for l in field])
        avg_salary = np.mean([l.total_salary for l in field])

        print("\nField Statistics:")
        print(f"  Avg Ownership: {avg_ownership:.1f}%")
        print(f"  Avg Salary: ${avg_salary:,.0f}")

        return field

    def calculate_lineup_uniqueness(self, lineup: FieldLineup, field: list[FieldLineup]) -> float:
        """Calculate how unique a lineup is compared to the field.
        
        Returns a score from 0 (completely duplicated) to 1 (completely unique).
        """
        if not field:
            return 1.0

        lineup_fighters = lineup.get_player_ids()
        overlap_scores = []

        for field_lineup in field:
            field_fighters = field_lineup.get_player_ids()
            overlap = len(lineup_fighters & field_fighters) / 6.0
            overlap_scores.append(overlap)

        # Average overlap with field
        avg_overlap = np.mean(overlap_scores)

        # Uniqueness is inverse of overlap
        uniqueness = 1.0 - avg_overlap

        return uniqueness

    def find_similar_lineups(self, lineup: FieldLineup, field: list[FieldLineup],
                           threshold: float = 0.5) -> list[tuple[FieldLineup, float]]:
        """Find field lineups similar to given lineup.
        
        Returns list of (lineup, similarity_score) tuples where similarity > threshold.
        """
        similar = []
        lineup_fighters = lineup.get_player_ids()

        for field_lineup in field:
            field_fighters = field_lineup.get_player_ids()
            overlap = len(lineup_fighters & field_fighters) / 6.0

            if overlap >= threshold:
                similar.append((field_lineup, overlap))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)

        return similar
