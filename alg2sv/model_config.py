"""
Model configuration for ARDA pipeline agents.
Defines which GPT model to use for each agent type.
"""

from typing import Any, Dict, Optional

# Model configuration mapping
AGENT_MODELS: Dict[str, str] = {
    # Use GPT-5 Mini for most agents (cost-effective, higher rate limits)
    'spec': 'gpt-5-nano',
    'quant': 'gpt-5-nano', 
    'microarch': 'gpt-5-nano',
    'static_checks': 'gpt-5-nano',
    'verification': 'gpt-5-nano',
    'evaluate': 'gpt-5-nano',
    'feedback': 'gpt-5-nano',
    
    # Use GPT-5 Pro for critical RTL generation (most complex task)
    'rtl': 'gpt-5-mini',
    
    # Synthesis agents can use Mini (they mainly call tools)
    'synth': 'gpt-5-nano',
}

# Fallback model if GPT-5 is not available
FALLBACK_MODEL = 'gpt-4.1'

def get_model_for_agent(agent_name: str) -> str:
    """Get the model name for a specific agent."""
    return AGENT_MODELS.get(agent_name, FALLBACK_MODEL)

def get_model_params_for_agent(agent_name: str) -> Dict[str, Any]:
    """Get model parameters for a specific agent."""
    model = get_model_for_agent(agent_name)
    params = {"model": model}
    
    # Note: reasoning_effort is not supported by the Responses API
    # Only add parameters that are supported by client.responses.create()
    
    return params

def is_gpt5_available() -> bool:
    """Check if GPT-5 models are available."""
    # This would need to be implemented based on actual API availability
    # For now, assume GPT-5 is available
    return True

def get_effective_model(agent_name: str) -> str:
    """Get the effective model to use, with fallback if GPT-5 not available."""
    if is_gpt5_available():
        return get_model_for_agent(agent_name)
    else:
        return FALLBACK_MODEL

# Default FPGA target
DEFAULT_FPGA_DEVICE = "xc7a100tcsg324-1"  # Nexys 7
DEFAULT_FPGA_FAMILY = "artix7"
DEFAULT_SYNTHESIS_BACKEND = "vivado"

# Device capabilities database
DEVICE_CAPABILITIES = {
    'xc7a100tcsg324-1': {
        'lut': 63400, 'ff': 126800, 'dsp': 240,
        'bram': 135, 'max_freq_mhz': 450,
        'typical_freq_mhz': 250, 'family': 'artix7'
    },
    'xc7a35tcpg236-1': {
        'lut': 20800, 'ff': 41600, 'dsp': 90,
        'bram': 50, 'max_freq_mhz': 400,
        'typical_freq_mhz': 200, 'family': 'artix7'
    },
    'ice40hx8k': {
        'lut': 7680, 'ff': 7680, 'dsp': 0,
        'bram': 32, 'max_freq_mhz': 50,
        'typical_freq_mhz': 25, 'family': 'ice40'
    }
}
