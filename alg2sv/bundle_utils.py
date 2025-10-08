"""
Bundle creation utilities for ARDA.

Converts Python files and directories into ARDA bundle format for easy ingestion.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any


class BundleCreator:
    """Creates ARDA bundles from Python files and directories."""

    def __init__(self):
        self.algorithm_patterns = [
            r'def step\s*\(',  # step() method
            r'def process\s*\(',  # process() method
            r'class.*Filter',  # Filter classes
            r'class.*Algorithm',  # Algorithm classes
            r'class.*DSP',  # DSP classes
        ]

    def create_bundle_from_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert a single Python file to bundle format.

        Args:
            file_path: Path to Python file
            output_path: Optional output path for bundle file

        Returns:
            Bundle content as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')

        if not file_path.endswith('.py'):
            raise ValueError('Only Python files (.py) are supported')

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create bundle format
        rel_path = os.path.basename(file_path)
        bundle_lines = [f'path={rel_path}']

        # Add algorithm metadata if detectable
        metadata = self._extract_algorithm_metadata(content)
        if metadata:
            bundle_lines.extend(metadata)

        bundle_lines.append(content)

        bundle_content = '\n'.join(bundle_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bundle_content)
            print(f'✅ Bundle created: {output_path}')

        return bundle_content

    def create_bundle_from_directory(self, dir_path: str, output_path: Optional[str] = None) -> str:
        """
        Scan directory for Python files and create bundle.

        Args:
            dir_path: Directory to scan
            output_path: Optional output path for bundle file

        Returns:
            Bundle content as string
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f'Directory not found: {dir_path}')

        bundle_lines = []

        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        if not python_files:
            raise ValueError(f'No Python files found in {dir_path}')

        # Sort files for consistent ordering
        python_files.sort()

        for file_path in python_files:
            rel_path = os.path.relpath(file_path, dir_path)
            bundle_lines.append(f'path={rel_path}')

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add algorithm metadata
            metadata = self._extract_algorithm_metadata(content)
            if metadata:
                bundle_lines.extend(metadata)

            bundle_lines.append(content)
            bundle_lines.append('')  # Separator between files

        bundle_content = '\n'.join(bundle_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(bundle_content)
            print(f'✅ Bundle created from {len(python_files)} files: {output_path}')

        return bundle_content

    def _extract_algorithm_metadata(self, content: str) -> List[str]:
        """Extract algorithm metadata from file content."""
        metadata = []

        # Check for algorithm patterns
        has_algorithm = any(re.search(pattern, content, re.MULTILINE)
                          for pattern in self.algorithm_patterns)

        if has_algorithm:
            metadata.append('# Auto-detected algorithm file')

            # Look for step() method
            step_match = re.search(r'def step\s*\([^)]*\)\s*:([^}]+)', content, re.DOTALL)
            if step_match:
                step_body = step_match.group(1).strip()
                if len(step_body) > 200:
                    step_body = step_body[:200] + '...'
                metadata.append(f'# Algorithm interface: step() method found')
                metadata.append(f'# Step method preview: {step_body}')

        return metadata


def create_bundle(source: str, output: Optional[str] = None) -> str:
    """
    Create bundle from file or directory path.

    Args:
        source: File path or directory path
        output: Optional output bundle file path

    Returns:
        Bundle content as string
    """
    creator = BundleCreator()

    if os.path.isfile(source):
        return creator.create_bundle_from_file(source, output)
    elif os.path.isdir(source):
        return creator.create_bundle_from_directory(source, output)
    else:
        raise ValueError(f'Path does not exist: {source}')


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python bundle_utils.py <file_or_directory> [output_file]')
        sys.exit(1)

    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        bundle = create_bundle(source, output)
        if not output:
            print('\nGenerated bundle:')
            print('=' * 50)
            print(bundle)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)
