"""Field generation for opponent lineup modeling in GPPs."""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Set, Tuple
from dataclasses import dataclass
from ..models.player import Player


@dataclass
class FieldLineup:
    """Represents an opponent's lineup in the field."""
    players: List[Player]  # Renamed from fighters to be sport-agnostic
    total_salary: int = 0
    total_ownership: float = 0
    lineup_id: int = 0
    
    def __post_init__(self):
        self.total_salary = sum(p.salary for p in self.players)
        self.total_ownership = sum(p.ownership for p in self.players)
    
    def get_player_ids(self) -> Set[int]:
        """Return set of player IDs for overlap calculation."""
        return {p.player_id for p in self.players}


class BaseFieldGenerator(ABC):
    """Abstract base class for sport-specific field generators."""
    
    def __init__(self, players: List[Player]):
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
    def _has_invalid_combinations(self, players: List[Player]) -> bool:
        """Check if lineup has sport-specific invalid combinations."""
        pass
    
    @abstractmethod
    def _select_by_ownership_probability(self, available: List[Player], n: int = 1) -> List[Player]:
        """Select players weighted by ownership probability."""
        pass
    
    @abstractmethod
    def generate_field(self, n_lineups: int = 10000) -> List[FieldLineup]:
        """Generate realistic field of opponent lineups."""
        pass


class MMAFieldGenerator(BaseFieldGenerator):
    """Generate realistic MMA opponent lineups based on ownership probabilities."""
    
    def __init__(self, players: List[Player]):
        super().__init__(players)
        self.salary_cap = 50000
        self.roster_size = 6
        
        # Pre-calculate opponent pairs to avoid
        self.opponent_pairs = self._build_opponent_pairs()
    
    def _build_opponent_pairs(self) -> Set[Tuple[int, int]]:
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
    
    
    def _has_invalid_combinations(self, players: List[Player]) -> bool:
        """Check if lineup contains opponents (MMA-specific rule)."""
        player_ids = {p.player_id for p in players}
        
        for p1_id in player_ids:
            for p2_id in player_ids:
                if p1_id != p2_id and (p1_id, p2_id) in self.opponent_pairs:
                    return True
        return False
    
    def _select_by_ownership_probability(self, available: List[Player], n: int = 1) -> List[Player]:
        """Select players weighted by ownership probability."""
        if not available or n <= 0:
            return []
        
        # Normalize ownership to probabilities
        ownerships = np.array([p.ownership for p in available])
        
        # Add small epsilon to avoid zero probability
        ownerships = np.maximum(ownerships, 0.1)
        
        # Convert to selection probabilities
        probabilities = ownerships / ownerships.sum()
        
        # Select without replacement
        n = min(n, len(available))
        selected_indices = np.random.choice(
            len(available), 
            size=n, 
            replace=False, 
            p=probabilities
        )
        
        return [available[i] for i in selected_indices]
    
    def generate_chalk_lineup(self) -> FieldLineup:
        """Generate a typical chalk-heavy lineup that most opponents will play."""
        lineup = []
        used_ids = set()
        attempts = 0
        max_attempts = 100
        
        while len(lineup) < 6 and attempts < max_attempts:
            attempts += 1
            
            # Chalk lineups: 3-4 chalk, 1-2 popular, 0-1 contrarian
            remaining = 6 - len(lineup)
            
            if remaining >= 4 and len(lineup) < 3:
                # Add chalk plays
                candidates = [p for p in self.chalk if p.player_id not in used_ids]
                if candidates:
                    selected = self._select_by_ownership_probability(candidates, 1)
                    if selected:
                        fighter = selected[0]
                        if not self._has_opponents(lineup + [fighter]):
                            lineup.append(fighter)
                            used_ids.add(fighter.player_id)
            
            elif remaining >= 2 and len(lineup) < 5:
                # Add popular plays
                candidates = [p for p in self.popular + self.moderate if p.player_id not in used_ids]
                if candidates:
                    selected = self._select_by_ownership_probability(candidates, 1)
                    if selected:
                        fighter = selected[0]
                        if not self._has_opponents(lineup + [fighter]):
                            lineup.append(fighter)
                            used_ids.add(fighter.player_id)
            
            else:
                # Fill with any valid fighter
                candidates = [p for p in self.players if p.player_id not in used_ids]
                if candidates:
                    selected = random.choice(candidates)
                    if not self._has_opponents(lineup + [selected]):
                        lineup.append(selected)
                        used_ids.add(selected.player_id)
        
        # If we couldn't build a full lineup, fill with random valid fighters
        if len(lineup) < 6:
            candidates = [p for p in self.players if p.player_id not in used_ids]
            for fighter in candidates:
                if len(lineup) >= 6:
                    break
                if not self._has_opponents(lineup + [fighter]):
                    lineup.append(fighter)
                    if len(lineup) == 6:
                        break
        
        return FieldLineup(fighters=lineup[:6])
    
    def generate_balanced_lineup(self) -> FieldLineup:
        """Generate a balanced lineup with mix of ownership levels."""
        lineup = []
        used_ids = set()
        
        # Target composition: 2 chalk/popular, 2 moderate, 2 contrarian/leverage
        ownership_targets = [
            (self.chalk + self.popular, 2),
            (self.moderate, 2),
            (self.contrarian + self.leverage, 2)
        ]
        
        for pool, target_count in ownership_targets:
            candidates = [p for p in pool if p.player_id not in used_ids]
            
            added = 0
            for _ in range(target_count):
                if not candidates:
                    break
                    
                selected = self._select_by_ownership_probability(candidates, 1)
                if selected:
                    fighter = selected[0]
                    if not self._has_opponents(lineup + [fighter]):
                        lineup.append(fighter)
                        used_ids.add(fighter.player_id)
                        candidates.remove(fighter)
                        added += 1
        
        # Fill remaining spots
        while len(lineup) < 6:
            candidates = [p for p in self.players if p.player_id not in used_ids]
            if not candidates:
                break
            
            fighter = random.choice(candidates)
            if not self._has_opponents(lineup + [fighter]):
                lineup.append(fighter)
                used_ids.add(fighter.player_id)
        
        return FieldLineup(fighters=lineup[:6])
    
    def generate_contrarian_lineup(self) -> FieldLineup:
        """Generate a contrarian lineup with mostly low-owned plays."""
        lineup = []
        used_ids = set()
        
        # Contrarian: 0-1 chalk, 1-2 moderate, 3-4 contrarian/leverage
        pools_priority = [
            (self.leverage, 2),
            (self.contrarian, 2),
            (self.moderate, 1),
            (self.popular, 1)
        ]
        
        for pool, target in pools_priority:
            candidates = [p for p in pool if p.player_id not in used_ids]
            
            for _ in range(target):
                if not candidates or len(lineup) >= 6:
                    break
                    
                fighter = random.choice(candidates)
                if not self._has_opponents(lineup + [fighter]):
                    lineup.append(fighter)
                    used_ids.add(fighter.player_id)
                    candidates.remove(fighter)
        
        # Fill remaining
        while len(lineup) < 6:
            candidates = [p for p in self.players if p.player_id not in used_ids]
            if not candidates:
                break
                
            fighter = self._select_by_ownership_probability(candidates, 1)[0]
            if not self._has_opponents(lineup + [fighter]):
                lineup.append(fighter)
                used_ids.add(fighter.player_id)
        
        return FieldLineup(fighters=lineup[:6])
    
    def generate_field(self, n_lineups: int = 10000) -> List[FieldLineup]:
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
        
        print(f"\nField Statistics:")
        print(f"  Avg Ownership: {avg_ownership:.1f}%")
        print(f"  Avg Salary: ${avg_salary:,.0f}")
        
        return field
    
    def calculate_lineup_uniqueness(self, lineup: FieldLineup, field: List[FieldLineup]) -> float:
        """Calculate how unique a lineup is compared to the field.
        
        Returns a score from 0 (completely duplicated) to 1 (completely unique).
        """
        if not field:
            return 1.0
        
        lineup_fighters = lineup.get_fighter_ids()
        overlap_scores = []
        
        for field_lineup in field:
            field_fighters = field_lineup.get_fighter_ids()
            overlap = len(lineup_fighters & field_fighters) / 6.0
            overlap_scores.append(overlap)
        
        # Average overlap with field
        avg_overlap = np.mean(overlap_scores)
        
        # Uniqueness is inverse of overlap
        uniqueness = 1.0 - avg_overlap
        
        return uniqueness
    
    def find_similar_lineups(self, lineup: FieldLineup, field: List[FieldLineup], 
                           threshold: float = 0.5) -> List[Tuple[FieldLineup, float]]:
        """Find field lineups similar to given lineup.
        
        Returns list of (lineup, similarity_score) tuples where similarity > threshold.
        """
        similar = []
        lineup_fighters = lineup.get_fighter_ids()
        
        for field_lineup in field:
            field_fighters = field_lineup.get_fighter_ids()
            overlap = len(lineup_fighters & field_fighters) / 6.0
            
            if overlap >= threshold:
                similar.append((field_lineup, overlap))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar