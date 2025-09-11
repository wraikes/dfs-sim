"""Contest model for DFS simulation."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ContestType(Enum):
    """DFS contest types."""
    GPP = "GPP"  # Guaranteed Prize Pool (tournament)
    CASH = "CASH"  # 50/50, Double-ups
    QUALIFIER = "QUALIFIER"  # Satellite to bigger contest
    SHOWDOWN = "SHOWDOWN"  # Single game contest


@dataclass
class Contest:
    """Represents a DFS contest with payout structure."""
    
    contest_id: str
    name: str
    contest_type: ContestType
    entry_fee: float
    total_entries: int
    max_entries_per_user: int
    prize_pool: float
    payout_structure: List[float]  # Percentage payouts by position
    
    # Optional fields
    sport: Optional[str] = None
    slate_id: Optional[str] = None
    start_time: Optional[str] = None
    
    @property
    def places_paid(self) -> int:
        """Get number of places that cash."""
        return len(self.payout_structure)
    
    @property
    def cash_line(self) -> float:
        """Get percentile needed to cash (0-100)."""
        return (self.places_paid / self.total_entries) * 100
    
    @property
    def first_place_prize(self) -> float:
        """Get first place prize amount."""
        if self.payout_structure:
            return self.prize_pool * self.payout_structure[0]
        return 0
    
    @property
    def min_cash_prize(self) -> float:
        """Get minimum cash prize amount."""
        if self.payout_structure:
            return self.prize_pool * self.payout_structure[-1]
        return 0
    
    @property
    def roi_at_position(self, position: int) -> float:
        """Calculate ROI for finishing at given position."""
        if position <= 0 or position > len(self.payout_structure):
            return -1.0  # Lost entry fee
        
        prize = self.prize_pool * self.payout_structure[position - 1]
        return (prize - self.entry_fee) / self.entry_fee
    
    def expected_value(self, win_probability: float, cash_probability: float) -> float:
        """
        Calculate expected value given win and cash probabilities.
        
        Args:
            win_probability: Probability of winning (0-1)
            cash_probability: Probability of cashing (0-1)
            
        Returns:
            Expected value in dollars
        """
        # Simplified EV calculation
        avg_cash_prize = sum(
            self.prize_pool * payout for payout in self.payout_structure
        ) / len(self.payout_structure)
        
        expected_return = (
            win_probability * self.first_place_prize +
            (cash_probability - win_probability) * avg_cash_prize
        )
        
        return expected_return - self.entry_fee
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Contest({self.name}, {self.contest_type.value}, "
            f"${self.entry_fee}, {self.total_entries} entries, "
            f"${self.prize_pool:,.0f} prize pool)"
        )