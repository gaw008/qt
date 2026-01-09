"""
Event Queue Builder

Identifies and prioritizes stocks for LLM analysis based on triggers and events.

Triggers:
- Earnings window (within 7 days of earnings date)
- Volume spike (unusual volume activity)
- Edge case (low confidence, score near threshold)
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def build_event_queue(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build event queue from selection results.

    Identifies stocks that have triggered events (earnings, volume spike, edge case).

    Args:
        results: List of selection results

    Returns:
        List of events with:
            - symbol: str
            - triggers: List[str] (list of triggered event types)
            - priority: int (higher = more important)
            - metadata: Dict (trigger-specific data)
    """
    events = []

    for result in results:
        symbol = result.get("symbol", "")
        if not symbol:
            continue

        triggers = []
        metadata = {}

        # Trigger 1: Earnings Window
        if _check_earnings_window(result):
            triggers.append("earnings_window")
            metadata["earnings_window"] = True

        # Trigger 2: Volume Spike
        volume_spike = _check_volume_spike(result)
        if volume_spike:
            triggers.append("volume_spike")
            metadata["volume_spike"] = volume_spike

        # Trigger 3: Edge Case
        edge_case = _check_edge_case(result)
        if edge_case:
            triggers.append("edge_case")
            metadata["edge_case"] = edge_case

        # If any triggers, add to queue
        if triggers:
            priority = _calculate_priority(triggers, result)
            events.append({
                "symbol": symbol,
                "triggers": triggers,
                "priority": priority,
                "metadata": metadata
            })

    logger.info(f"[LLM] Built event queue with {len(events)} triggered stocks")
    return events


def prioritize(events: List[Dict[str, Any]]) -> List[str]:
    """
    Prioritize events and return sorted list of symbols.

    Args:
        events: List of event dicts from build_event_queue

    Returns:
        List of symbols sorted by priority (highest first)
    """
    # Sort by priority descending
    sorted_events = sorted(events, key=lambda x: x.get("priority", 0), reverse=True)

    # Return just symbols
    return [event["symbol"] for event in sorted_events]


def _check_earnings_window(result: Dict[str, Any]) -> bool:
    """
    Check if stock is in earnings window (within 7 days of earnings date).

    Args:
        result: Selection result

    Returns:
        True if in earnings window
    """
    # Check if earnings_date is available in metadata
    earnings_date_str = result.get("metadata", {}).get("earnings_date")
    if not earnings_date_str:
        return False

    try:
        earnings_date = datetime.fromisoformat(earnings_date_str)
        now = datetime.now()

        # Within 7 days before or after earnings
        days_diff = abs((earnings_date - now).days)
        if days_diff <= 7:
            logger.debug(f"[LLM] {result['symbol']}: Earnings window trigger (days={days_diff})")
            return True

    except Exception as e:
        logger.debug(f"[LLM] Error parsing earnings date: {e}")

    return False


def _check_volume_spike(result: Dict[str, Any]) -> Optional[float]:
    """
    Check for volume spike.

    Args:
        result: Selection result

    Returns:
        Volume spike ratio (current/average) or None if no spike
    """
    # Get volume data from strategies metadata
    strategies = result.get("strategies", {})
    metadata = result.get("metadata", {})

    current_volume = metadata.get("current_volume")
    avg_volume = metadata.get("avg_volume")

    if current_volume and avg_volume and avg_volume > 0:
        ratio = current_volume / avg_volume

        # Spike if volume > 2x average
        if ratio > 2.0:
            logger.debug(f"[LLM] {result['symbol']}: Volume spike trigger (ratio={ratio:.2f})")
            return ratio

    return None


def _check_edge_case(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check if stock is an edge case (low confidence or near threshold).

    Args:
        result: Selection result

    Returns:
        Edge case details or None
    """
    symbol = result.get("symbol", "")
    avg_score = result.get("avg_score", 0)
    strategies = result.get("strategies", {})

    edge_cases = []

    # Edge Case 1: Low overall score (50-60 range - borderline)
    if 50 <= avg_score <= 60:
        edge_cases.append("borderline_score")

    # Edge Case 2: High variance across strategies
    if len(strategies) >= 2:
        scores = list(strategies.values())
        score_variance = max(scores) - min(scores)
        if score_variance > 30:
            edge_cases.append("high_variance")

    # Edge Case 3: Technical Breakout with low score
    tb_score = strategies.get("technical_breakout", 0)
    if tb_score > 0 and tb_score < 55:
        edge_cases.append("weak_technical_breakout")

    if edge_cases:
        logger.debug(f"[LLM] {symbol}: Edge case trigger ({', '.join(edge_cases)})")
        return {
            "cases": edge_cases,
            "avg_score": avg_score,
            "variance": max(strategies.values()) - min(strategies.values()) if len(strategies) >= 2 else 0
        }

    return None


def _calculate_priority(triggers: List[str], result: Dict[str, Any]) -> int:
    """
    Calculate priority score for event.

    Higher priority = more important to analyze.

    Args:
        triggers: List of triggered event types
        result: Selection result

    Returns:
        Priority score (0-100)
    """
    priority = 0

    # Base priority from number of triggers
    priority += len(triggers) * 10

    # Boost for specific triggers
    if "earnings_window" in triggers:
        priority += 30  # Highest priority

    if "volume_spike" in triggers:
        priority += 20

    if "edge_case" in triggers:
        priority += 15

    # Boost based on base score (higher score = higher priority)
    avg_score = result.get("avg_score", 0)
    priority += int(avg_score * 0.2)  # Max +20 points

    # Cap at 100
    return min(priority, 100)
