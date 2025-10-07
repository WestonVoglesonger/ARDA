"""
Virtual Workspace Management for ALG2SV Pipeline
Handles algorithm bundles and generated artifacts in memory.
"""

import re
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class FileItem(BaseModel):
    """Represents a file in the workspace."""
    path: str
    content: str


class Workspace:
    """Virtual workspace for storing algorithm files and generated artifacts."""

    def __init__(self):
        self.files: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}

    def add_file(self, path: str, content: str) -> None:
        """Add or update a file in the workspace."""
        normalized_path = self._normalize_path(path)
        self.files[normalized_path] = content

    def get_file(self, path: str) -> Optional[str]:
        """Get file content by path."""
        normalized_path = self._normalize_path(path)
        return self.files.get(normalized_path)

    def list_files(self) -> List[str]:
        """List all files in the workspace."""
        return sorted(self.files.keys())

    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        normalized_path = self._normalize_path(path)
        return normalized_path in self.files

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize file paths."""
        return path.replace("\\", "/").strip()


class WorkspaceManager:
    """Manages multiple workspaces for concurrent pipeline runs."""

    def __init__(self):
        self.workspaces: Dict[str, Workspace] = {}

    def create_workspace(self) -> str:
        """Create a new workspace and return its ID."""
        workspace_id = str(uuid.uuid4())
        self.workspaces[workspace_id] = Workspace()
        return workspace_id

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        return self.workspaces.get(workspace_id)

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace."""
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]
            return True
        return False

    def ingest_bundle(self, raw_bundle: str, normalize_paths: bool = False) -> str:
        """
        Parse a bundle string and create a workspace.

        Bundle format:
        ``` path=file1.py
        content here
        ```

        ``` path=file2.yaml
        content here
        ```
        """
        # Regex to match file blocks
        file_block_pattern = r'```(?:file)?\s*path\s*=\s*([^\n]+)\n([\s\S]*?)```'

        workspace_id = self.create_workspace()
        workspace = self.get_workspace(workspace_id)

        for match in re.finditer(file_block_pattern, raw_bundle):
            path = match.group(1).strip()
            content = match.group(2).rstrip()

            # Remove trailing fence if present
            content = re.sub(r'```\s*$', '', content).rstrip()

            if normalize_paths:
                path = path.lower()

            workspace.add_file(path, content)

        return workspace_id


# Global workspace manager instance
workspace_manager = WorkspaceManager()


def ingest_from_bundle(raw_bundle: str, normalize: bool = False) -> Dict[str, Any]:
    """
    Tool function: Ingest algorithm bundle into workspace.

    Args:
        raw_bundle: String containing multiple files in fence format
        normalize: Whether to normalize paths to lowercase

    Returns:
        Dict with workspace_id, file count, and paths
    """
    try:
        workspace_id = workspace_manager.ingest_bundle(raw_bundle, normalize)
        workspace = workspace_manager.get_workspace(workspace_id)

        return {
            "workspace_token": workspace_id,  # Use workspace_id as token
            "count": len(workspace.files),
            "paths": workspace.list_files(),
            "success": True
        }
    except Exception as e:
        return {
            "error": f"Failed to parse bundle: {str(e)}",
            "success": False
        }


def read_source(workspace_token: str, path: str) -> Dict[str, Any]:
    """
    Tool function: Read a file from the workspace.

    Args:
        workspace_token: Workspace identifier
        path: File path to read

    Returns:
        Dict with file content or error
    """
    try:
        workspace = workspace_manager.get_workspace(workspace_token)
        if not workspace:
            return {
                "error": "Workspace not found",
                "success": False
            }

        content = workspace.get_file(path)
        if content is None:
            return {
                "error": f"File not found: {path}",
                "available_paths": workspace.list_files(),
                "success": False
            }

        return {
            "path": path,
            "content": content,
            "size": len(content),
            "success": True
        }
    except Exception as e:
        return {
            "error": f"Failed to read file: {str(e)}",
            "success": False
        }


def write_artifact(workspace_token: str, path: str, content: str) -> Dict[str, Any]:
    """
    Tool function: Write a file to the workspace.

    Args:
        workspace_token: Workspace identifier
        path: File path to write
        content: File content

    Returns:
        Dict with success status
    """
    try:
        workspace = workspace_manager.get_workspace(workspace_token)
        if not workspace:
            return {
                "error": "Workspace not found",
                "success": False
            }

        previous_size = len(workspace.get_file(path) or "")
        workspace.add_file(path, content)

        return {
            "workspace_token": workspace_token,
            "path": path,
            "size": len(content),
            "previous_size": previous_size,
            "success": True
        }
    except Exception as e:
        return {
            "error": f"Failed to write file: {str(e)}",
            "success": False
        }
