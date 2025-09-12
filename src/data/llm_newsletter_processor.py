"""LLM-based newsletter processor for DFS player adjustments."""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from ..models import Player

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class PlayerSignal:
    """Represents an LLM-extracted signal for a player."""
    name: str
    signal: str  # 'target', 'avoid', 'neutral'
    confidence: float  # 0.0 to 1.0
    ownership_delta: float  # -0.10 to +0.10
    ceiling_delta: float  # -0.12 to +0.12
    reason: str  # Explanation from newsletter
    

@dataclass
class NewsletterExtraction:
    """Complete newsletter extraction result."""
    players: List[PlayerSignal]
    global_context: Dict[str, Any]
    strategic_notes: List[str]
    

class LLMNewsletterProcessor:
    """Process newsletter content using LLM to extract structured signals."""
    
    def __init__(self, sport: str, model: str = "gpt-4o-mini"):
        """
        Initialize LLM processor.
        
        Args:
            sport: Sport name (mma, nascar, nfl)
            model: OpenAI model to use
        """
        self.sport = sport.lower()
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Sport-specific contexts for better extraction
        self.sport_contexts = {
            'mma': {
                'position_types': ['Fighter'],
                'key_stats': ['takedowns', 'strikes', 'submissions', 'finish rate'],
                'value_indicators': ['grappling', 'striking', 'cardio', 'power']
            },
            'nascar': {
                'position_types': ['Driver'], 
                'key_stats': ['qualifying', 'practice speed', 'track position'],
                'value_indicators': ['speed', 'track history', 'equipment']
            },
            'nfl': {
                'position_types': ['QB', 'RB', 'WR', 'TE', 'DST'],
                'key_stats': ['targets', 'carries', 'snaps', 'red zone'],
                'value_indicators': ['matchup', 'usage', 'game script']
            }
        }
    
    def extract_signals(self, newsletter_text: str) -> NewsletterExtraction:
        """
        Extract structured signals from newsletter text using LLM.
        
        Args:
            newsletter_text: Raw newsletter content
            
        Returns:
            NewsletterExtraction with structured signals
        """
        system_prompt = self._build_system_prompt()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": newsletter_text}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle potential markdown formatting)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.rfind("```")
                content = content[json_start:json_end].strip()
            
            data = json.loads(content)
            return self._parse_extraction_data(data)
            
        except Exception as e:
            logger.error(f"Error extracting newsletter signals: {e}")
            return NewsletterExtraction(players=[], global_context={}, strategic_notes=[])
    
    def _build_system_prompt(self) -> str:
        """Build sport-specific system prompt for LLM."""
        context = self.sport_contexts.get(self.sport, {})
        
        return f'''You are a DFS Newsletter Signal Extractor for {self.sport.upper()}.

INPUT: Raw DFS newsletter text analyzing players and matchups.
OUTPUT: Structured JSON following this EXACT schema:

{{
  "players": [
    {{
      "name": "Player Full Name",
      "signal": "target|avoid|neutral", 
      "confidence": 0.3-0.9,
      "reason": "Brief explanation from newsletter"
    }}
  ],
  "global_context": {{
    "weather": "conditions if mentioned",
    "slate_size": "number if mentioned", 
    "key_factors": ["factor1", "factor2"]
  }},
  "strategic_notes": ["note1", "note2"]
}}

EXTRACTION RULES:
1. NEVER invent players not mentioned in the text
2. Set confidence based on tone strength (this will scale the deltas):
   - Strong emphasis/locks ("smash spot", "must play", "lock"): 0.8-0.9
   - Clear recommendation ("strong play", "like a lot", "target"): 0.6-0.8  
   - Moderate mention ("decent option", "consider"): 0.4-0.6
   - Mild mention/balanced view ("could work", "maybe"): 0.3-0.4

3. Base delta directions by signal type:
   - POSITIVE (target): ceiling_delta = +0.15 * confidence, ownership_delta = -0.12 * confidence  
   - NEGATIVE (avoid): ceiling_delta = -0.15 * confidence, ownership_delta = +0.12 * confidence

4. Sport context for {self.sport.upper()}:
   - Positions: {context.get('position_types', ['Player'])}
   - Key stats: {context.get('key_stats', ['performance metrics'])}
   - Value indicators: {context.get('value_indicators', ['matchup factors'])}

5. CLAMP all deltas within allowed ranges
6. Extract strategic themes into global_context and strategic_notes
7. Return ONLY valid JSON, no extra text

Focus on extracting actionable DFS signals, not general game analysis.'''

    def _parse_extraction_data(self, data: Dict) -> NewsletterExtraction:
        """Parse LLM response data into NewsletterExtraction."""
        players = []
        
        for p in data.get('players', []):
            # Validate confidence
            confidence = max(0.0, min(1.0, float(p.get('confidence', 0.5))))
            signal_type = p.get('signal', 'neutral').lower()
            
            # Calculate deltas based on confidence and signal type (aggressive weighting)
            if signal_type == 'target':
                ceiling_delta = 0.15 * confidence  # Max +0.135 at 0.9 confidence
                ownership_delta = -0.12 * confidence  # Max -0.108 at 0.9 confidence  
            elif signal_type == 'avoid':
                ceiling_delta = -0.15 * confidence  # Max -0.135 at 0.9 confidence
                ownership_delta = 0.12 * confidence  # Max +0.108 at 0.9 confidence
            else:
                ceiling_delta = 0.0
                ownership_delta = 0.0
            
            # Clamp to safe ranges
            ownership_delta = max(-0.10, min(0.10, ownership_delta))
            ceiling_delta = max(-0.12, min(0.12, ceiling_delta))
            
            signal = PlayerSignal(
                name=p.get('name', '').strip(),
                signal=signal_type,
                confidence=confidence,
                ownership_delta=ownership_delta,
                ceiling_delta=ceiling_delta,
                reason=p.get('reason', '')
            )
            players.append(signal)
        
        return NewsletterExtraction(
            players=players,
            global_context=data.get('global_context', {}),
            strategic_notes=data.get('strategic_notes', [])
        )
    
    def apply_signals_to_players(self, players: List[Player], extraction: NewsletterExtraction) -> List[Player]:
        """
        Apply extracted signals to player objects.
        
        Args:
            players: List of Player objects
            extraction: NewsletterExtraction with signals
            
        Returns:
            List of Player objects with signals applied
        """
        # Create lookup map for efficient matching
        signal_map = {}
        for signal in extraction.players:
            # Store both exact name and normalized name
            exact_key = signal.name.lower().strip()
            signal_map[exact_key] = signal
            
            # Also store lastname-firstname combinations for better matching
            parts = signal.name.split()
            if len(parts) >= 2:
                lastname_first = f"{parts[-1]} {' '.join(parts[:-1])}".lower().strip()
                signal_map[lastname_first] = signal
        
        modified_count = 0
        for player in players:
            player_key = player.name.lower().strip()
            
            # Try exact match
            signal = signal_map.get(player_key)
            
            # Try partial matching if no exact match
            if not signal:
                for signal_name, potential_signal in signal_map.items():
                    if self._names_match(player.name, potential_signal.name):
                        signal = potential_signal
                        break
            
            if signal:
                # Apply ceiling adjustment
                if signal.ceiling_delta != 0:
                    multiplier = 1.0 + signal.ceiling_delta
                    player.ceiling = max(player.projection, player.ceiling * multiplier)
                
                # Store ownership adjustment for later use
                if signal.ownership_delta != 0:
                    # Adjust ownership (will be used by simulation engine)
                    new_ownership = max(0, min(100, player.ownership + (signal.ownership_delta * 100)))
                    player.ownership = new_ownership
                
                # Apply target/fade multipliers (aggressive weighting)
                if signal.signal == 'target':
                    player.target_multiplier = 1.0 + (signal.confidence * 0.4)  # Up to 40% boost
                elif signal.signal == 'avoid':
                    player.fade_multiplier = 1.0 - (signal.confidence * 0.4)  # Up to 40% reduction
                
                # Mark as volatile if high confidence target (affects std_dev)
                if signal.signal == 'target' and signal.confidence > 0.7:
                    player.volatile = True
                
                # Store signal metadata
                player.metadata['newsletter_signal'] = signal.signal
                player.metadata['newsletter_confidence'] = signal.confidence
                player.metadata['newsletter_reason'] = signal.reason
                player.metadata['ceiling_delta'] = signal.ceiling_delta
                player.metadata['ownership_delta'] = signal.ownership_delta
                
                modified_count += 1
        
        logger.info(f"Applied newsletter signals to {modified_count} players")
        return players
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same player."""
        n1_parts = set(name1.lower().split())
        n2_parts = set(name2.lower().split())
        
        # If names share at least 2 words (first + last name typically)
        if len(n1_parts & n2_parts) >= 2:
            return True
        
        # Check if one name is contained in the other
        if len(n1_parts) >= 2 and len(n2_parts) >= 2:
            n1_clean = name1.lower().replace('.', '').replace(',', '')
            n2_clean = name2.lower().replace('.', '').replace(',', '')
            
            return n1_clean in n2_clean or n2_clean in n1_clean
        
        return False
    
    def process_newsletter_file(self, filepath: str, save_json: bool = True) -> NewsletterExtraction:
        """
        Process a newsletter file and extract signals.
        
        Args:
            filepath: Path to newsletter file (should be in /newsletters/ folder)
            save_json: Whether to save JSON signals to /json/ folder
            
        Returns:
            NewsletterExtraction with signals
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Newsletter file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        extraction = self.extract_signals(content)
        
        # Auto-save JSON signals to correct folder if requested
        if save_json and extraction.players:
            json_path = self._get_json_output_path(filepath)
            self.save_signals_json(extraction, json_path)
        
        return extraction
    
    def _get_json_output_path(self, newsletter_path: str) -> str:
        """Get the correct JSON output path based on newsletter path."""
        path = Path(newsletter_path)
        
        # Find the base data directory (e.g., data/mma/466/)
        parts = path.parts
        if 'newsletters' in parts:
            # Get everything up to newsletters folder
            newsletter_idx = parts.index('newsletters')
            base_parts = parts[:newsletter_idx]
            
            # Create json path
            json_dir = Path(*base_parts) / 'json'
            
            # Use newsletter filename but change extension
            filename = path.stem + '_signals.json'
            return str(json_dir / filename)
        
        # Fallback to same directory
        return str(path.parent / f"{path.stem}_signals.json")
    
    def save_signals_json(self, extraction: NewsletterExtraction, output_path: str) -> str:
        """
        Save extraction results to JSON file.
        
        Args:
            extraction: NewsletterExtraction to save
            output_path: Path to save JSON file
            
        Returns:
            Path where JSON was saved
        """
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert extraction to dict
        json_data = {
            'players': [
                {
                    'name': p.name,
                    'signal': p.signal, 
                    'confidence': p.confidence,
                    'ownership_delta': p.ownership_delta,
                    'ceiling_delta': p.ceiling_delta,
                    'reason': p.reason
                }
                for p in extraction.players
            ],
            'global_context': extraction.global_context,
            'strategic_notes': extraction.strategic_notes
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Saved newsletter signals to: {output_path}")
        return str(output_path)
    
    def get_summary(self, extraction: NewsletterExtraction) -> str:
        """Get human-readable summary of extraction."""
        if not extraction.players:
            return "No newsletter signals extracted"
        
        summary = f"Newsletter Analysis Summary ({len(extraction.players)} players):\n\n"
        
        # Group by signal type
        targets = [p for p in extraction.players if p.signal == 'target']
        avoids = [p for p in extraction.players if p.signal == 'avoid']
        
        if targets:
            summary += f"TARGETS ({len(targets)}):\n"
            for p in sorted(targets, key=lambda x: x.confidence, reverse=True):
                summary += f"  • {p.name} (confidence: {p.confidence:.2f})\n"
                summary += f"    └─ {p.reason}\n"
        
        if avoids:
            summary += f"\nAVOIDS ({len(avoids)}):\n"
            for p in sorted(avoids, key=lambda x: x.confidence, reverse=True):
                summary += f"  • {p.name} (confidence: {p.confidence:.2f})\n"
                summary += f"    └─ {p.reason}\n"
        
        if extraction.strategic_notes:
            summary += f"\nSTRATEGIC NOTES:\n"
            for note in extraction.strategic_notes:
                summary += f"  • {note}\n"
        
        return summary