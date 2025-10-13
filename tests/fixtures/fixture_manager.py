"""
Fixture manager for OpenAI stage tests.

Manages loading and validation of stage fixtures, providing realistic test data
for dependency injection in stage tests.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tests.utils.schema_validator import SchemaValidator, SchemaValidationError


class FixtureManager:
    """Manages loading and validation of stage fixtures."""
    
    def __init__(self, fixtures_dir: Optional[str] = None):
        """
        Initialize fixture manager.
        
        Args:
            fixtures_dir: Directory containing fixture files (default: tests/fixtures)
        """
        self.fixtures_dir = Path(fixtures_dir or "tests/fixtures")
        self.schema_validator = SchemaValidator()
        self._fixture_cache = {}
    
    def load_algorithm_fixtures(self, algorithm: str) -> Dict[str, Any]:
        """
        Load all fixtures for a specific algorithm.
        
        Args:
            algorithm: Name of the algorithm (e.g., 'conv2d')
            
        Returns:
            Dictionary mapping stage names to fixture data
            
        Raises:
            FileNotFoundError: If fixture file doesn't exist
            ValueError: If fixture data is invalid
        """
        if algorithm in self._fixture_cache:
            return self._fixture_cache[algorithm]
        
        fixture_file = self.fixtures_dir / algorithm / f"{algorithm}_fixtures.json"
        
        if not fixture_file.exists():
            raise FileNotFoundError(f"Fixture file not found: {fixture_file}")
        
        try:
            with open(fixture_file, 'r', encoding='utf-8') as f:
                fixtures = json.load(f)
            
            # Validate fixtures against schemas
            self._validate_fixtures(algorithm, fixtures)
            
            # Cache the validated fixtures
            self._fixture_cache[algorithm] = fixtures
            
            return fixtures
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in fixture file {fixture_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading fixtures for {algorithm}: {e}")
    
    def load_stage_fixture(self, algorithm: str, stage: str) -> Any:
        """
        Load fixture data for a specific algorithm and stage.
        
        Args:
            algorithm: Name of the algorithm
            stage: Name of the stage
            
        Returns:
            Fixture data for the specified stage
            
        Raises:
            KeyError: If stage fixture doesn't exist
        """
        fixtures = self.load_algorithm_fixtures(algorithm)
        
        if stage not in fixtures:
            available_stages = list(fixtures.keys())
            raise KeyError(f"Stage '{stage}' not found in {algorithm} fixtures. Available: {available_stages}")
        
        return fixtures[stage]
    
    def load_bundle_fixture(self, algorithm: str) -> str:
        """
        Load bundle fixture for a specific algorithm.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Bundle content as string
            
        Raises:
            FileNotFoundError: If bundle file doesn't exist
        """
        bundle_file = self.fixtures_dir / algorithm / f"{algorithm}_bundle.txt"
        
        if not bundle_file.exists():
            raise FileNotFoundError(f"Bundle file not found: {bundle_file}")
        
        with open(bundle_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_stage_dependencies(self, algorithm: str, stage: str) -> Dict[str, Any]:
        """
        Get dependency fixtures for a stage.
        
        Args:
            algorithm: Name of the algorithm
            stage: Name of the stage
            
        Returns:
            Dictionary mapping dependency stage names to their fixture data
        """
        fixtures = self.load_algorithm_fixtures(algorithm)
        
        # Define stage dependencies
        dependencies = {
            "spec": [],
            "quant": ["spec"],
            "microarch": ["spec"],
            "architecture": ["spec", "quant", "microarch"],
            "rtl": ["spec", "quant", "microarch", "architecture"],
            "verification": ["spec", "quant", "microarch", "architecture", "rtl"],
            "synth": ["spec", "quant", "microarch", "architecture", "rtl", "verification"],
            "evaluate": ["spec", "quant", "microarch", "architecture", "rtl", "verification", "synth"]
        }
        
        stage_deps = dependencies.get(stage, [])
        dep_fixtures = {}
        
        for dep_stage in stage_deps:
            if dep_stage in fixtures:
                dep_fixtures[dep_stage] = fixtures[dep_stage]
        
        return dep_fixtures
    
    def _validate_fixtures(self, algorithm: str, fixtures: Dict[str, Any]) -> None:
        """
        Validate fixtures against expected schemas.
        
        Args:
            algorithm: Name of the algorithm
            fixtures: Fixture data to validate
            
        Raises:
            SchemaValidationError: If validation fails
        """
        errors = []
        
        for stage, fixture_data in fixtures.items():
            try:
                # Validate against schema
                validated = self.schema_validator.validate_stage_output(stage, fixture_data)
                
                # Check confidence score if present
                if isinstance(fixture_data, dict) and "confidence" in fixture_data:
                    if not self.schema_validator.validate_confidence_score(stage, fixture_data, 70.0):
                        errors.append(f"{algorithm}.{stage}: Low confidence score ({fixture_data.get('confidence')})")
                
            except SchemaValidationError as e:
                errors.append(f"{algorithm}.{stage}: {str(e)}")
            except Exception as e:
                errors.append(f"{algorithm}.{stage}: Unexpected error: {e}")
        
        if errors:
            error_msg = "Fixture validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise SchemaValidationError(error_msg)
    
    def list_available_algorithms(self) -> List[str]:
        """
        List all available algorithms with fixtures.
        
        Returns:
            List of algorithm names
        """
        algorithms = []
        
        if not self.fixtures_dir.exists():
            return algorithms
        
        for algorithm_dir in self.fixtures_dir.iterdir():
            if algorithm_dir.is_dir():
                fixture_file = algorithm_dir / f"{algorithm_dir.name}_fixtures.json"
                if fixture_file.exists():
                    algorithms.append(algorithm_dir.name)
        
        return sorted(algorithms)
    
    def list_available_stages(self, algorithm: str) -> List[str]:
        """
        List all available stages for an algorithm.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            List of stage names
        """
        try:
            fixtures = self.load_algorithm_fixtures(algorithm)
            return list(fixtures.keys())
        except (FileNotFoundError, ValueError):
            return []
    
    def get_fixture_info(self, algorithm: str) -> Dict[str, Any]:
        """
        Get information about fixtures for an algorithm.
        
        Args:
            algorithm: Name of the algorithm
            
        Returns:
            Dictionary with fixture information
        """
        try:
            fixtures = self.load_algorithm_fixtures(algorithm)
            
            info = {
                "algorithm": algorithm,
                "stages": list(fixtures.keys()),
                "has_bundle": (self.fixtures_dir / algorithm / f"{algorithm}_bundle.txt").exists(),
                "total_stages": len(fixtures),
                "stage_details": {}
            }
            
            for stage, data in fixtures.items():
                info["stage_details"][stage] = {
                    "has_confidence": "confidence" in data if isinstance(data, dict) else False,
                    "confidence_score": data.get("confidence") if isinstance(data, dict) else None,
                    "data_type": type(data).__name__
                }
            
            return info
            
        except (FileNotFoundError, ValueError) as e:
            return {
                "algorithm": algorithm,
                "error": str(e),
                "stages": [],
                "has_bundle": False,
                "total_stages": 0,
                "stage_details": {}
            }
    
    def clear_cache(self) -> None:
        """Clear the fixture cache."""
        self._fixture_cache.clear()
    
    def validate_all_fixtures(self) -> Dict[str, List[str]]:
        """
        Validate all available fixtures.
        
        Returns:
            Dictionary mapping algorithm names to lists of validation errors
        """
        results = {}
        algorithms = self.list_available_algorithms()
        
        for algorithm in algorithms:
            try:
                self.load_algorithm_fixtures(algorithm)
                results[algorithm] = []  # No errors
            except Exception as e:
                results[algorithm] = [str(e)]
        
        return results
