# backward compat shim — file moved to app.core.agent.quant_agent
from backend.app.agent.quant_agent import (  # noqa: F401
    QuantAgent, ConversationMemory, FallbackOrchestrator,
    QuantTools, OverfitCritic,
)
