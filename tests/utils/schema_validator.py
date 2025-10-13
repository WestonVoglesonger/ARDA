"""
Schema validator for OpenAI stage outputs.

Validates stage outputs against Pydantic models to ensure correct structure
and required fields are present.
"""

from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ValidationError

from ardagen.domain import (
    SpecContract,
    QuantConfig,
    MicroArchConfig,
    ArchitectureConfig,
    RTLConfig,
    VerifyResults,
    SynthResults,
    EvaluateResults,
)


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class SchemaValidator:
    """Validates OpenAI stage outputs against expected Pydantic schemas."""
    
    # Map stage names to their expected output models
    STAGE_MODELS: Dict[str, Type[BaseModel]] = {
        "spec": SpecContract,
        "quant": QuantConfig,
        "microarch": MicroArchConfig,
        "architecture": ArchitectureConfig,
        "rtl": RTLConfig,
        "verification": VerifyResults,
        "synth": SynthResults,
        "evaluate": EvaluateResults,
    }
    
    def __init__(self):
        """Initialize schema validator."""
        pass
    
    def validate_stage_output(
        self,
        stage: str,
        output: Any,
        strict: bool = True
    ) -> BaseModel:
        """
        Validate stage output against expected schema.
        
        Args:
            stage: Name of the stage (e.g., 'spec', 'quant')
            output: Raw output from OpenAI agent
            strict: If True, require exact schema match. If False, allow extra fields.
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            SchemaValidationError: If validation fails
        """
        if stage not in self.STAGE_MODELS:
            raise SchemaValidationError(f"Unknown stage: {stage}")
        
        model_class = self.STAGE_MODELS[stage]
        
        try:
            # Convert output to dict if it's not already
            if isinstance(output, dict):
                output_dict = output
            elif hasattr(output, 'model_dump'):
                output_dict = output.model_dump()
            elif hasattr(output, 'dict'):
                output_dict = output.dict()
            else:
                raise SchemaValidationError(f"Cannot convert output to dict: {type(output)}")
            
            # Validate against Pydantic model
            if strict:
                validated = model_class(**output_dict)
            else:
                # Allow extra fields by using model_validate with extra='ignore'
                validated = model_class.model_validate(output_dict, strict=False)
            
            return validated
            
        except ValidationError as e:
            error_details = self._format_validation_error(e)
            raise SchemaValidationError(
                f"Schema validation failed for stage '{stage}':\n{error_details}"
            ) from e
        except Exception as e:
            raise SchemaValidationError(
                f"Unexpected error validating stage '{stage}': {e}"
            ) from e
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """Format Pydantic validation error for human readability."""
        errors = []
        
        for err in error.errors():
            field_path = " -> ".join(str(loc) for loc in err["loc"])
            error_type = err["type"]
            error_msg = err["msg"]
            input_value = err.get("input", "N/A")
            
            errors.append(
                f"  Field '{field_path}': {error_msg} (type: {error_type}, value: {input_value})"
            )
        
        return "\n".join(errors)
    
    def validate_required_fields(
        self,
        stage: str,
        output: Dict[str, Any]
    ) -> List[str]:
        """
        Check if all required fields are present in output.
        
        Args:
            stage: Name of the stage
            output: Output dictionary to check
            
        Returns:
            List of missing required field names
        """
        if stage not in self.STAGE_MODELS:
            return [f"Unknown stage: {stage}"]
        
        model_class = self.STAGE_MODELS[stage]
        missing_fields = []
        
        # Get required fields from model
        required_fields = set()
        for field_name, field_info in model_class.model_fields.items():
            if field_info.is_required():
                required_fields.add(field_name)
        
        # Check for missing fields
        for field_name in required_fields:
            if field_name not in output:
                missing_fields.append(field_name)
        
        return missing_fields
    
    def get_expected_schema(self, stage: str) -> Optional[Dict[str, Any]]:
        """
        Get expected schema for a stage.
        
        Args:
            stage: Name of the stage
            
        Returns:
            Dictionary describing the expected schema, or None if stage unknown
        """
        if stage not in self.STAGE_MODELS:
            return None
        
        model_class = self.STAGE_MODELS[stage]
        
        schema = {
            "model_name": model_class.__name__,
            "fields": {},
            "required_fields": []
        }
        
        for field_name, field_info in model_class.model_fields.items():
            field_schema = {
                "type": str(field_info.annotation),
                "description": field_info.description or "No description",
                "default": field_info.default if field_info.default is not None else "No default"
            }
            
            schema["fields"][field_name] = field_schema
            
            if field_info.is_required():
                schema["required_fields"].append(field_name)
        
        return schema
    
    def validate_confidence_score(
        self,
        stage: str,
        output: Dict[str, Any],
        min_confidence: float = 70.0
    ) -> bool:
        """
        Validate that confidence score meets minimum threshold.
        
        Args:
            stage: Name of the stage
            output: Output dictionary
            min_confidence: Minimum acceptable confidence score
            
        Returns:
            True if confidence is acceptable, False otherwise
        """
        confidence = output.get("confidence")
        
        if confidence is None:
            # Some stages might not have confidence scores
            return True
        
        if not isinstance(confidence, (int, float)):
            return False
        
        return confidence >= min_confidence
