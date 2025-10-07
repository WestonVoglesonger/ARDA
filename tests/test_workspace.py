"""
Tests for workspace management functionality.
"""

import pytest
from alg2sv.workspace import ingest_from_bundle, read_source, write_artifact, WorkspaceManager


class TestWorkspace:
    """Test workspace functionality."""

    def test_ingest_bundle(self):
        """Test bundle ingestion."""
        bundle = """``` path=test.py
def hello():
    return "world"
```

``` path=config.yaml
name: test
value: 42
```"""

        result = ingest_from_bundle(bundle)

        assert result['success'] == True
        assert 'workspace_token' in result
        assert result['count'] == 2
        assert 'test.py' in result['paths']
        assert 'config.yaml' in result['paths']

    def test_read_write_operations(self):
        """Test read/write operations."""
        # Create workspace
        bundle = """``` path=test.txt
Hello World
```"""
        ingest_result = ingest_from_bundle(bundle)
        workspace_token = ingest_result['workspace_token']

        # Read file
        read_result = read_source(workspace_token, 'test.txt')
        assert read_result['success'] == True
        assert read_result['content'] == 'Hello World'

        # Write file
        write_result = write_artifact(workspace_token, 'output.txt', 'New Content')
        assert write_result['success'] == True

        # Read new file
        read_result2 = read_source(workspace_token, 'output.txt')
        assert read_result2['success'] == True
        assert read_result2['content'] == 'New Content'

    def test_bundle_parsing_edge_cases(self):
        """Test edge cases in bundle parsing."""
        # Empty bundle
        result = ingest_from_bundle("")
        assert result['success'] == True
        assert result['count'] == 0

        # Malformed fence
        malformed = """``` path=test.py
content
```missing"""
        result = ingest_from_bundle(malformed)
        assert result['success'] == True
        assert result['count'] == 1

        # Valid bundle with extra content
        valid = """Some text before
``` path=test.py
content here
```
Some text after"""
        result = ingest_from_bundle(valid)
        assert result['success'] == True
        assert result['count'] == 1
