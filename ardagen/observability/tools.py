"""
Observability and Performance Monitoring Tools for the ARDA Pipeline.
Provides distributed tracing, performance monitoring, and error tracking.
"""

import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from agents import function_tool

# Global tracing state
_trace_data = []
_trace_lock = threading.Lock()


@dataclass
class TraceEvent:
    """Trace event for distributed tracing."""
    timestamp: float
    agent_name: str
    stage: str
    event_type: str  # 'start', 'end', 'tool_call', 'error'
    event_data: Dict[str, Any]
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None


@function_tool
def trace_logger_tool(agent_name: str, stage: str, event_type: str, event_data: str) -> str:
    """
    OpenTelemetry-style distributed tracing of agent decisions.
    
    Args:
        agent_name: Name of the agent
        stage: Pipeline stage
        event_type: Type of event (start, end, tool_call, error)
        event_data: Additional event data
        
    Returns:
        JSON with trace confirmation
    """
    try:
        # Parse event_data JSON string
        event_data_dict = json.loads(event_data)
        
        with _trace_lock:
            trace_event = TraceEvent(
                timestamp=time.time(),
                agent_name=agent_name,
                stage=stage,
                event_type=event_type,
                event_data=event_data_dict,
                trace_id=f"{agent_name}_{stage}_{int(time.time())}"
            )
            
            _trace_data.append(trace_event)
            
            # Keep only last 1000 events to prevent memory issues
            if len(_trace_data) > 1000:
                _trace_data.pop(0)
        
        return json.dumps({
            'success': True,
            'trace_id': trace_event.trace_id,
            'timestamp': trace_event.timestamp,
            'event_count': len(_trace_data)
        })
        
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})


@function_tool
def performance_monitor_tool(agent_name: str, metrics: str) -> str:
    """
    Track token usage, latency, quality metrics per stage.
    
    Args:
        agent_name: Name of the agent
        metrics: Performance metrics to track
        
    Returns:
        JSON with performance summary
    """
    try:
        # Parse metrics JSON string
        metrics_dict = json.loads(metrics)
        
        # Extract key metrics
        input_tokens = metrics_dict.get('input_tokens', 0)
        output_tokens = metrics_dict.get('output_tokens', 0)
        latency_ms = metrics_dict.get('latency_ms', 0)
        quality_score = metrics_dict.get('quality_score', 0)
        
        # Calculate derived metrics
        total_tokens = input_tokens + output_tokens
        tokens_per_second = total_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        
        # Store performance data
        perf_data = {
            'timestamp': time.time(),
            'agent_name': agent_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'latency_ms': latency_ms,
            'tokens_per_second': tokens_per_second,
            'quality_score': quality_score,
            'cost_estimate': _estimate_cost(input_tokens, output_tokens)
        }
        
        # Log performance data
        logging.info(f"Performance: {agent_name} - {total_tokens} tokens, {latency_ms}ms, score: {quality_score}")
        
        return json.dumps({
            'success': True,
            'performance_data': perf_data,
            'recommendations': _get_performance_recommendations(perf_data)
        })
        
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})


@function_tool
def error_tracker_tool(error_type: str, error_message: str, context: str) -> str:
    """
    Log and categorize errors with root cause analysis.
    
    Args:
        error_type: Type of error (syntax, semantic, timing, etc.)
        error_message: Error message
        context: Additional context information
        
    Returns:
        JSON with error analysis and suggestions
    """
    try:
        # Parse context JSON string
        context_dict = json.loads(context)
        
        error_data = {
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context_dict,
            'severity': _classify_error_severity(error_type, error_message),
            'root_cause': _analyze_root_cause(error_type, error_message),
            'suggestions': _get_error_suggestions(error_type, error_message)
        }
        
        # Log error
        logging.error(f"Error [{error_type}]: {error_message}")
        
        return json.dumps({
            'success': True,
            'error_data': error_data,
            'error_id': f"ERR_{int(time.time())}_{error_type.upper()}"
        })
        
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})


@function_tool
def visualization_tool(visualization_type: str, data: str) -> str:
    """
    Generate diagrams (architecture, dataflow, timing) for evaluation.
    
    Args:
        visualization_type: Type of visualization (architecture, dataflow, timing, performance)
        data: Data to visualize
        
    Returns:
        JSON with visualization data and instructions
    """
    try:
        # Parse data JSON string
        data_dict = json.loads(data)
        
        if visualization_type == 'architecture':
            return _generate_architecture_diagram(data_dict)
        elif visualization_type == 'dataflow':
            return _generate_dataflow_diagram(data_dict)
        elif visualization_type == 'timing':
            return _generate_timing_diagram(data_dict)
        elif visualization_type == 'performance':
            return _generate_performance_chart(data_dict)
        else:
            return json.dumps({
                'success': False,
                'error': f'Unknown visualization type: {visualization_type}'
            })
            
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})


@function_tool
def get_trace_summary_tool() -> str:
    """
    Get summary of all trace events for debugging and analysis.
    
    Returns:
        JSON with trace summary and statistics
    """
    try:
        with _trace_lock:
            if not _trace_data:
                return json.dumps({
                    'success': True,
                    'message': 'No trace data available',
                    'trace_count': 0
                })
            
            # Calculate statistics
            agent_counts = {}
            stage_counts = {}
            event_type_counts = {}
            total_duration = 0
            
            for event in _trace_data:
                agent_counts[event.agent_name] = agent_counts.get(event.agent_name, 0) + 1
                stage_counts[event.stage] = stage_counts.get(event.stage, 0) + 1
                event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
                
                if event.duration_ms:
                    total_duration += event.duration_ms
            
            # Get recent events
            recent_events = _trace_data[-10:] if len(_trace_data) >= 10 else _trace_data
            
            return json.dumps({
                'success': True,
                'trace_summary': {
                    'total_events': len(_trace_data),
                    'agent_counts': agent_counts,
                    'stage_counts': stage_counts,
                    'event_type_counts': event_type_counts,
                    'total_duration_ms': total_duration,
                    'recent_events': [asdict(event) for event in recent_events]
                }
            })
            
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})


def _estimate_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    """Estimate cost based on token usage."""
    # Rough cost estimates (per 1M tokens)
    gpt4_input_cost = 30.0  # $30 per 1M input tokens
    gpt4_output_cost = 60.0  # $60 per 1M output tokens
    gpt5_mini_input_cost = 1.25  # $1.25 per 1M input tokens
    gpt5_mini_output_cost = 10.0  # $10 per 1M output tokens
    
    gpt4_cost = (input_tokens * gpt4_input_cost + output_tokens * gpt4_output_cost) / 1_000_000
    gpt5_mini_cost = (input_tokens * gpt5_mini_input_cost + output_tokens * gpt5_mini_output_cost) / 1_000_000
    
    return {
        'gpt4_cost': round(gpt4_cost, 6),
        'gpt5_mini_cost': round(gpt5_mini_cost, 6),
        'savings': round(gpt4_cost - gpt5_mini_cost, 6)
    }


def _get_performance_recommendations(perf_data: Dict[str, Any]) -> List[str]:
    """Get performance recommendations based on metrics."""
    recommendations = []
    
    if perf_data['latency_ms'] > 30000:  # 30 seconds
        recommendations.append("High latency detected - consider optimizing prompts or using faster model")
    
    if perf_data['tokens_per_second'] < 10:
        recommendations.append("Low token throughput - check for rate limiting or network issues")
    
    if perf_data['quality_score'] < 70:
        recommendations.append("Low quality score - review agent instructions and tool usage")
    
    if perf_data['total_tokens'] > 50000:
        recommendations.append("High token usage - consider caching or reducing context size")
    
    return recommendations


def _classify_error_severity(error_type: str, error_message: str) -> str:
    """Classify error severity."""
    critical_keywords = ['fatal', 'critical', 'abort', 'timeout', 'crash']
    warning_keywords = ['warning', 'deprecated', 'minor', 'style']
    
    error_text = (error_type + ' ' + error_message).lower()
    
    if any(keyword in error_text for keyword in critical_keywords):
        return 'critical'
    elif any(keyword in error_text for keyword in warning_keywords):
        return 'warning'
    else:
        return 'error'


def _analyze_root_cause(error_type: str, error_message: str) -> str:
    """Analyze root cause of error."""
    error_text = (error_type + ' ' + error_message).lower()
    
    if 'syntax' in error_text:
        return 'Syntax error in generated code'
    elif 'timing' in error_text:
        return 'Timing constraint violation'
    elif 'resource' in error_text:
        return 'Resource usage exceeds limits'
    elif 'verification' in error_text:
        return 'Verification test failure'
    elif 'rate limit' in error_text:
        return 'API rate limit exceeded'
    else:
        return 'Unknown root cause - requires investigation'


def _get_error_suggestions(error_type: str, error_message: str) -> List[str]:
    """Get suggestions for fixing errors."""
    error_text = (error_type + ' ' + error_message).lower()
    
    suggestions = []
    
    if 'syntax' in error_text:
        suggestions.extend([
            'Check SystemVerilog syntax',
            'Verify module instantiation',
            'Check port connections'
        ])
    elif 'timing' in error_text:
        suggestions.extend([
            'Increase pipeline depth',
            'Reduce critical path delay',
            'Add pipeline registers'
        ])
    elif 'resource' in error_text:
        suggestions.extend([
            'Optimize resource usage',
            'Consider different implementation',
            'Reduce parallelism'
        ])
    elif 'rate limit' in error_text:
        suggestions.extend([
            'Implement exponential backoff',
            'Reduce request frequency',
            'Use caching for repeated requests'
        ])
    
    return suggestions


def _generate_architecture_diagram(data: Dict[str, Any]) -> str:
    """Generate architecture diagram."""
    return json.dumps({
        'success': True,
        'visualization_type': 'architecture',
        'diagram_data': {
            'type': 'mermaid',
            'content': f"""
graph TD
    A[Algorithm] --> B[Spec Agent]
    B --> C[Quant Agent]
    C --> D[MicroArch Agent]
    D --> E[RTL Agent]
    E --> F[Synth Agent]
    F --> G[Lint Agent]
    G --> H[Evaluate Agent]
""",
            'instructions': [
                'Use Mermaid.js to render the diagram',
                'Shows pipeline flow and dependencies',
                'Can be embedded in documentation'
            ]
        }
    })


def _generate_dataflow_diagram(data: Dict[str, Any]) -> str:
    """Generate dataflow diagram."""
    return json.dumps({
        'success': True,
        'visualization_type': 'dataflow',
        'diagram_data': {
            'type': 'mermaid',
            'content': """
graph LR
    A[Input Data] --> B[Processing]
    B --> C[Pipeline Stage 1]
    C --> D[Pipeline Stage 2]
    D --> E[Output Data]
""",
            'instructions': [
                'Shows data flow through pipeline',
                'Useful for understanding bottlenecks',
                'Can be customized with actual data paths'
            ]
        }
    })


def _generate_timing_diagram(data: Dict[str, Any]) -> str:
    """Generate timing diagram."""
    return json.dumps({
        'success': True,
        'visualization_type': 'timing',
        'diagram_data': {
            'type': 'mermaid',
            'content': """
gantt
    title Pipeline Execution Timeline
    dateFormat X
    axisFormat %s
    
    section Agents
    Spec Agent    :0, 5
    Quant Agent  :5, 10
    MicroArch    :10, 15
    RTL Agent    :15, 30
    Synth Agent  :30, 45
    Lint Agent   :45, 50
    Evaluate    :50, 55
""",
            'instructions': [
                'Shows timing relationships',
                'Useful for performance analysis',
                'Can show parallel execution'
            ]
        }
    })


def _generate_performance_chart(data: Dict[str, Any]) -> str:
    """Generate performance chart."""
    return json.dumps({
        'success': True,
        'visualization_type': 'performance',
        'chart_data': {
            'type': 'line_chart',
            'data': {
                'labels': ['Spec', 'Quant', 'MicroArch', 'RTL', 'Synth', 'Lint', 'Evaluate'],
                'datasets': [{
                    'label': 'Latency (ms)',
                    'data': [100, 200, 150, 1000, 2000, 300, 400],
                    'borderColor': 'rgb(75, 192, 192)'
                }]
            },
            'options': {
                'responsive': True,
                'title': {
                    'display': True,
                    'text': 'Agent Performance Metrics'
                }
            }
        },
        'instructions': [
            'Use Chart.js to render the chart',
            'Shows performance metrics over time',
            'Can be customized with actual data'
        ]
    })
