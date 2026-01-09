#!/usr/bin/env python3
"""
AI Integration Extension for Tiger Data Provider
Adds AI-powered trading decision support to the existing Tiger Data Provider
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Import existing modules
try:
    from real_ai_training_manager import real_ai_manager
    from real_ai_recommendations import get_real_ai_recommendations
    AI_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI integration modules not available: {e}")
    AI_INTEGRATION_AVAILABLE = False


class AIIntegrationMixin:
    """
    Mixin class to add AI capabilities to TigerDataProvider
    """

    async def get_ai_recommendations(self, symbols: List[str]) -> Dict:
        """Get AI-based trading recommendations for given symbols"""
        if not AI_INTEGRATION_AVAILABLE:
            return {"status": "ai_not_available", "recommendations": []}

        try:
            # Get training progress from AI manager
            training_progress = real_ai_manager.get_training_progress()

            # Generate recommendations based on AI training results
            recommendations = []
            for symbol in symbols:
                recommendation = await self._generate_ai_recommendation(symbol, training_progress)
                recommendations.append(recommendation)

            logger.info(f"AI recommendations obtained for {len(symbols)} symbols")
            return {
                "status": "success",
                "recommendations": recommendations,
                "ai_metrics": training_progress,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get AI recommendations: {e}")
            return {"status": "error", "error": str(e), "recommendations": []}

    async def _generate_ai_recommendation(self, symbol: str, training_progress: Dict) -> Dict:
        """Generate AI recommendation for a single symbol"""
        try:
            # Extract AI model metrics
            model_confidence = training_progress.get("validation_accuracy", 0.5)
            win_rate = training_progress.get("win_rate", 0.5)
            sharpe_ratio = training_progress.get("sharpe_ratio", 0.0)
            training_status = training_progress.get("status", "idle")

            # Determine action based on AI metrics
            if training_status == "completed" and sharpe_ratio > 1.0:
                if model_confidence > 0.8:
                    action = "buy"
                    confidence = min(0.9, model_confidence)
                elif model_confidence < 0.4:
                    action = "sell"
                    confidence = 0.7
                else:
                    action = "hold"
                    confidence = 0.6
            else:
                action = "hold"
                confidence = 0.5

            # Calculate risk level
            risk_score = 1.0 - min(model_confidence, win_rate)
            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"

            # Determine order optimization
            order_type = "limit" if confidence > 0.7 else "market"
            quantity_factor = max(0.5, min(1.5, confidence * 1.5))

            return {
                "symbol": symbol,
                "action": action,
                "confidence_score": confidence,
                "risk_level": risk_level,
                "recommended_order_type": order_type,
                "optimal_quantity_factor": quantity_factor,
                "timing_score": model_confidence * win_rate,
                "ai_metrics": {
                    "model_confidence": model_confidence,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "training_status": training_status
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to generate AI recommendation for {symbol}: {e}")
            return {
                "symbol": symbol,
                "action": "hold",
                "confidence_score": 0.5,
                "risk_level": "medium",
                "recommended_order_type": "limit",
                "optimal_quantity_factor": 1.0,
                "timing_score": 0.5,
                "ai_metrics": {},
                "timestamp": datetime.now().isoformat()
            }

    async def optimize_order_with_ai(self, order_request: Dict) -> Dict:
        """Optimize order using AI recommendations"""
        if not AI_INTEGRATION_AVAILABLE:
            return order_request

        try:
            symbol = order_request["symbol"]
            ai_recommendations = await self.get_ai_recommendations([symbol])

            if ai_recommendations["status"] == "success" and ai_recommendations["recommendations"]:
                recommendation = ai_recommendations["recommendations"][0]
                optimized_order = order_request.copy()

                # Apply AI optimization
                if recommendation.get("recommended_order_type"):
                    optimized_order["type"] = recommendation["recommended_order_type"]
                    logger.info(f"AI optimized order type for {symbol}: {recommendation['recommended_order_type']}")

                # Quantity optimization based on AI confidence
                if recommendation.get("optimal_quantity_factor"):
                    factor = recommendation["optimal_quantity_factor"]
                    if 0.5 <= factor <= 2.0:  # Safety bounds
                        optimized_order["quantity"] = int(order_request["quantity"] * factor)
                        logger.info(f"AI optimized quantity for {symbol}: {optimized_order['quantity']}")

                # Add AI metadata to order
                optimized_order["ai_recommendation"] = {
                    "action": recommendation["action"],
                    "confidence": recommendation["confidence_score"],
                    "risk_level": recommendation["risk_level"],
                    "timing_score": recommendation["timing_score"]
                }

                return optimized_order

            return order_request

        except Exception as e:
            logger.error(f"AI order optimization failed: {e}")
            return order_request

    async def get_ai_enhanced_assets(self, limit: int = 50, offset: int = 0, asset_type: Optional[str] = None) -> List[Dict]:
        """Get market assets enhanced with AI scoring"""
        try:
            # Get basic asset data
            assets = await self.get_assets(limit, offset, asset_type)

            if not AI_INTEGRATION_AVAILABLE:
                return assets

            # Extract symbols for AI analysis
            symbols = [asset["symbol"] for asset in assets]
            ai_recommendations = await self.get_ai_recommendations(symbols)

            if ai_recommendations["status"] == "success":
                # Enhance assets with AI scores
                ai_data = {rec["symbol"]: rec for rec in ai_recommendations["recommendations"]}

                for asset in assets:
                    symbol = asset["symbol"]
                    if symbol in ai_data:
                        ai_rec = ai_data[symbol]
                        asset["ai_score"] = ai_rec.get("confidence_score", 0)
                        asset["ai_recommendation"] = ai_rec.get("action", "hold")
                        asset["ai_risk_level"] = ai_rec.get("risk_level", "medium")
                        asset["ai_timing_score"] = ai_rec.get("timing_score", 0.5)

                # Sort by AI score if available
                assets.sort(key=lambda x: x.get("ai_score", 0), reverse=True)

            return assets

        except Exception as e:
            logger.error(f"AI asset enhancement failed: {e}")
            return assets

    def get_ai_training_status(self) -> Dict:
        """Get current AI training status"""
        if not AI_INTEGRATION_AVAILABLE:
            return {"status": "ai_not_available"}

        try:
            progress = real_ai_manager.get_training_progress()
            strategy_weights = real_ai_manager.get_strategy_weights()

            return {
                "status": "available",
                "training_progress": progress,
                "strategy_weights": strategy_weights,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get AI training status: {e}")
            return {"status": "error", "error": str(e)}


# Function to patch TigerDataProvider with AI capabilities
def add_ai_integration(tiger_provider_instance):
    """
    Add AI integration capabilities to an existing TigerDataProvider instance
    """
    if not AI_INTEGRATION_AVAILABLE:
        logger.warning("AI integration not available - skipping AI capability injection")
        return False

    try:
        # Add AI training manager
        tiger_provider_instance.ai_training_manager = real_ai_manager

        # Add AI methods to the instance
        import types

        mixin = AIIntegrationMixin()

        # Bind methods to the instance
        tiger_provider_instance.get_ai_recommendations = types.MethodType(
            AIIntegrationMixin.get_ai_recommendations, tiger_provider_instance
        )
        tiger_provider_instance._generate_ai_recommendation = types.MethodType(
            AIIntegrationMixin._generate_ai_recommendation, tiger_provider_instance
        )
        tiger_provider_instance.optimize_order_with_ai = types.MethodType(
            AIIntegrationMixin.optimize_order_with_ai, tiger_provider_instance
        )
        tiger_provider_instance.get_ai_enhanced_assets = types.MethodType(
            AIIntegrationMixin.get_ai_enhanced_assets, tiger_provider_instance
        )
        tiger_provider_instance.get_ai_training_status = types.MethodType(
            AIIntegrationMixin.get_ai_training_status, tiger_provider_instance
        )

        logger.info("AI integration capabilities added to TigerDataProvider")
        return True

    except Exception as e:
        logger.error(f"Failed to add AI integration: {e}")
        return False