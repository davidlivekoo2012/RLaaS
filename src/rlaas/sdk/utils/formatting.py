"""
Formatting utilities for RLaaS SDK.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
from ..models import APIError


def format_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format API response.
    
    Args:
        response: Raw API response
        
    Returns:
        Formatted response
    """
    # Handle error responses
    if "error" in response:
        return format_error_response(response)
    
    # Format timestamps
    formatted_response = {}
    for key, value in response.items():
        if key.endswith("_at") or key.endswith("_time"):
            formatted_response[key] = format_datetime(value)
        elif isinstance(value, dict):
            formatted_response[key] = format_response(value)
        elif isinstance(value, list):
            formatted_response[key] = [
                format_response(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            formatted_response[key] = value
    
    return formatted_response


def format_error_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format error response.
    
    Args:
        response: Error response
        
    Returns:
        Formatted error response
    """
    error_info = response.get("error", {})
    
    if isinstance(error_info, str):
        return {
            "error": {
                "code": "UNKNOWN_ERROR",
                "message": error_info,
                "details": {}
            }
        }
    
    return {
        "error": {
            "code": error_info.get("code", "UNKNOWN_ERROR"),
            "message": error_info.get("message", "An unknown error occurred"),
            "details": error_info.get("details", {})
        }
    }


def format_error(error: Exception) -> APIError:
    """
    Format exception as API error.
    
    Args:
        error: Exception to format
        
    Returns:
        API error object
    """
    if hasattr(error, 'error_code'):
        return APIError(
            error_code=error.error_code,
            message=str(error),
            details=getattr(error, 'details', None)
        )
    
    # Map common exception types
    error_type = type(error).__name__
    error_code_map = {
        "ConnectionError": "CONNECTION_ERROR",
        "TimeoutError": "TIMEOUT_ERROR",
        "HTTPError": "HTTP_ERROR",
        "ValueError": "VALIDATION_ERROR",
        "KeyError": "MISSING_FIELD_ERROR",
        "TypeError": "TYPE_ERROR",
        "FileNotFoundError": "FILE_NOT_FOUND",
        "PermissionError": "PERMISSION_DENIED"
    }
    
    error_code = error_code_map.get(error_type, "UNKNOWN_ERROR")
    
    return APIError(
        error_code=error_code,
        message=str(error),
        details={"exception_type": error_type}
    )


def format_datetime(value: Union[str, datetime, None]) -> Optional[str]:
    """
    Format datetime value.
    
    Args:
        value: Datetime value to format
        
    Returns:
        Formatted datetime string or None
    """
    if value is None:
        return None
    
    if isinstance(value, str):
        try:
            # Try to parse and reformat
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.isoformat()
        except ValueError:
            # Return as-is if parsing fails
            return value
    
    if isinstance(value, datetime):
        return value.isoformat()
    
    return str(value)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes in human-readable format.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    
    size = float(bytes_value)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage value.
    
    Args:
        value: Percentage value (0-1 or 0-100)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    # Assume 0-1 range if value is <= 1
    if value <= 1:
        percentage = value * 100
    else:
        percentage = value
    
    return f"{percentage:.{decimals}f}%"


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format number with appropriate precision.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if isinstance(value, int):
        return str(value)
    
    if abs(value) >= 1000000:
        return f"{value/1000000:.{decimals}f}M"
    elif abs(value) >= 1000:
        return f"{value/1000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def format_optimization_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format optimization result for display.
    
    Args:
        result: Raw optimization result
        
    Returns:
        Formatted result
    """
    formatted = result.copy()
    
    # Format execution time
    if "execution_time" in formatted:
        formatted["execution_time_formatted"] = format_duration(formatted["execution_time"])
    
    # Format best solution objectives
    if "best_solution" in formatted and formatted["best_solution"]:
        objectives = formatted["best_solution"].get("objectives", {})
        formatted_objectives = {}
        
        for obj_name, obj_value in objectives.items():
            if "latency" in obj_name.lower():
                formatted_objectives[obj_name] = f"{obj_value:.2f}ms"
            elif "throughput" in obj_name.lower():
                formatted_objectives[obj_name] = f"{obj_value:.2f}Mbps"
            elif "energy" in obj_name.lower():
                formatted_objectives[obj_name] = f"{obj_value:.3f}W"
            elif "rate" in obj_name.lower() or "ratio" in obj_name.lower():
                formatted_objectives[obj_name] = format_percentage(obj_value)
            else:
                formatted_objectives[obj_name] = format_number(obj_value)
        
        formatted["best_solution"]["objectives_formatted"] = formatted_objectives
    
    return formatted


def format_training_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format training result for display.
    
    Args:
        result: Raw training result
        
    Returns:
        Formatted result
    """
    formatted = result.copy()
    
    # Format training time
    if "training_time" in formatted:
        formatted["training_time_formatted"] = format_duration(formatted["training_time"])
    
    # Format final metrics
    if "final_metrics" in formatted and formatted["final_metrics"]:
        metrics = formatted["final_metrics"]
        formatted_metrics = {}
        
        for metric_name, metric_value in metrics.items():
            if metric_name == "loss":
                formatted_metrics[metric_name] = f"{metric_value:.6f}"
            elif metric_name in ["accuracy", "reward"]:
                formatted_metrics[metric_name] = f"{metric_value:.4f}"
            else:
                formatted_metrics[metric_name] = format_number(metric_value)
        
        formatted["final_metrics_formatted"] = formatted_metrics
    
    return formatted


def format_model_info(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format model information for display.
    
    Args:
        model: Raw model information
        
    Returns:
        Formatted model info
    """
    formatted = model.copy()
    
    # Format latency
    if "avg_latency_ms" in formatted:
        formatted["avg_latency_formatted"] = f"{formatted['avg_latency_ms']:.2f}ms"
    
    # Format error rate
    if "error_rate" in formatted:
        formatted["error_rate_formatted"] = format_percentage(formatted["error_rate"])
    
    # Format request count
    if "request_count" in formatted:
        formatted["request_count_formatted"] = format_number(formatted["request_count"], 0)
    
    return formatted


def format_dataset_info(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format dataset information for display.
    
    Args:
        dataset: Raw dataset information
        
    Returns:
        Formatted dataset info
    """
    formatted = dataset.copy()
    
    # Format size
    if "size_bytes" in formatted:
        formatted["size_formatted"] = format_bytes(formatted["size_bytes"])
    
    # Format row count
    if "row_count" in formatted:
        formatted["row_count_formatted"] = format_number(formatted["row_count"], 0)
    
    return formatted


def format_table(data: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> str:
    """
    Format data as ASCII table.
    
    Args:
        data: List of dictionaries to format
        columns: Columns to include (all if None)
        
    Returns:
        ASCII table string
    """
    if not data:
        return "No data to display"
    
    # Determine columns
    if columns is None:
        columns = list(data[0].keys())
    
    # Calculate column widths
    widths = {}
    for col in columns:
        widths[col] = max(
            len(str(col)),
            max(len(str(row.get(col, ""))) for row in data)
        )
    
    # Create header
    header = " | ".join(str(col).ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    
    # Create rows
    rows = []
    for row in data:
        formatted_row = " | ".join(
            str(row.get(col, "")).ljust(widths[col]) for col in columns
        )
        rows.append(formatted_row)
    
    return "\n".join([header, separator] + rows)


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format data as JSON string.
    
    Args:
        data: Data to format
        indent: JSON indentation
        
    Returns:
        JSON string
    """
    return json.dumps(data, indent=indent, default=str, ensure_ascii=False)
