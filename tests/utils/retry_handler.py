"""
Interactive retry handler for OpenAI test failures.

Handles test failures with user interaction, displaying failure reasons
and prompting for retry decisions.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class RetryHandler:
    """Handles interactive retries for failed OpenAI tests."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retries allowed
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def execute_with_retry(
        self,
        test_func: Callable[[], Any],
        test_name: str,
        algorithm: str,
        stage: str
    ) -> Tuple[Any, int, List[str]]:
        """
        Execute test function with interactive retry on failure.
        
        Args:
            test_func: Function to execute (should take no arguments)
            test_name: Name of the test for display
            algorithm: Algorithm being tested
            stage: Stage being tested
            
        Returns:
            Tuple of (result, retry_count, error_messages)
        """
        errors = []
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                result = test_func()
                return result, retry_count, errors
                
            except Exception as e:
                error_msg = str(e)
                errors.append(error_msg)
                
                if retry_count >= self.max_retries:
                    print(f"\n❌ Test '{test_name}' failed after {retry_count} retries")
                    print(f"   Algorithm: {algorithm}, Stage: {stage}")
                    print(f"   Final error: {error_msg}")
                    raise e
                
                # Ask user if they want to retry
                should_retry = self._prompt_for_retry(
                    test_name, algorithm, stage, error_msg, retry_count
                )
                
                if not should_retry:
                    print(f"\n🛑 Test '{test_name}' aborted by user")
                    raise e
                
                retry_count += 1
                print(f"\n🔄 Retrying test '{test_name}' (attempt {retry_count + 1}/{self.max_retries + 1})")
                
                if self.retry_delay > 0:
                    time.sleep(self.retry_delay)
        
        # Should never reach here, but just in case
        raise Exception(f"Unexpected retry loop exit for test '{test_name}'")
    
    def _prompt_for_retry(
        self,
        test_name: str,
        algorithm: str,
        stage: str,
        error_msg: str,
        retry_count: int
    ) -> bool:
        """
        Prompt user for retry decision.
        
        Args:
            test_name: Name of the test
            algorithm: Algorithm being tested
            stage: Stage being tested
            error_msg: Error message from the failure
            retry_count: Current retry count
            
        Returns:
            True if user wants to retry, False otherwise
        """
        print(f"\n⚠️  Test '{test_name}' failed")
        print(f"   Algorithm: {algorithm}")
        print(f"   Stage: {stage}")
        print(f"   Attempt: {retry_count + 1}/{self.max_retries + 1}")
        print(f"   Error: {error_msg}")
        
        # Categorize error for better user guidance
        error_category = self._categorize_error(error_msg)
        if error_category:
            print(f"   Category: {error_category}")
        
        # Check if we're in a pytest environment (stdin is captured)
        import sys
        import os
        
        # Detect pytest environment
        is_pytest = (
            'pytest' in sys.modules or 
            'pytest' in str(type(sys.stdin)) or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            hasattr(sys.stdin, 'read') and 'DontReadFromInput' in str(type(sys.stdin))
        )
        
        if is_pytest:
            auto_retry = self.should_auto_retry(Exception(error_msg))
            print(f"\n   Auto-retry (pytest mode): {auto_retry}")
            return auto_retry
        
        while True:
            try:
                response = input(f"\n   Retry? [y/n]: ").strip().lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("   Please enter 'y' or 'n'")
            except (KeyboardInterrupt, OSError):
                print("\n   Auto-retry based on error type")
                return self.should_auto_retry(Exception(error_msg))
    
    def _categorize_error(self, error_msg: str) -> Optional[str]:
        """
        Categorize error message for better user guidance.
        
        Args:
            error_msg: Error message to categorize
            
        Returns:
            Error category string or None
        """
        error_lower = error_msg.lower()
        
        if any(keyword in error_lower for keyword in ['rate limit', 'quota', 'limit exceeded']):
            return "Rate Limit - Wait before retrying"
        elif any(keyword in error_lower for keyword in ['timeout', 'timed out', 'connection']):
            return "Network Issue - Retry likely to succeed"
        elif any(keyword in error_lower for keyword in ['validation', 'schema', 'invalid']):
            return "Schema Error - Check agent configuration"
        elif any(keyword in error_lower for keyword in ['api key', 'authentication', 'unauthorized']):
            return "Authentication Error - Check API key"
        elif any(keyword in error_lower for keyword in ['model', 'not found', 'invalid model']):
            return "Model Error - Check model configuration"
        else:
            return "Unknown Error - Retry may or may not help"
    
    def handle_openai_error(self, error: Exception) -> str:
        """
        Handle OpenAI-specific errors and provide user-friendly messages.
        
        Args:
            error: Exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_str = str(error)
        
        # Common OpenAI error patterns
        if "rate_limit_exceeded" in error_str:
            return "OpenAI API rate limit exceeded. Please wait before retrying."
        elif "insufficient_quota" in error_str:
            return "OpenAI API quota exceeded. Check your billing settings."
        elif "invalid_api_key" in error_str:
            return "Invalid OpenAI API key. Check your .env file."
        elif "timeout" in error_str:
            return "Request timed out. This may be a temporary network issue."
        elif "connection" in error_str:
            return "Connection error. Check your internet connection."
        else:
            return f"OpenAI API error: {error_str}"
    
    def should_auto_retry(self, error: Exception) -> bool:
        """
        Determine if an error should be automatically retried without user prompt.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if error should be auto-retried
        """
        error_str = str(error).lower()
        
        # Auto-retry for transient errors
        auto_retry_patterns = [
            'rate_limit_exceeded',
            'timeout',
            'connection',
            'temporary',
            'service unavailable'
        ]
        
        return any(pattern in error_str for pattern in auto_retry_patterns)
