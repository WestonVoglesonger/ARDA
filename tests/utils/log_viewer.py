#!/usr/bin/env python3
"""
Log viewer utility for OpenAI stage tests.

Provides easy navigation and summary of test logs organized by stage and status.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import argparse


def get_log_summary(log_dir: Path) -> Dict[str, Any]:
    """Get comprehensive summary of all test logs."""
    summary = {
        "total_tests": 0,
        "by_stage": {},
        "by_status": {"passed": 0, "failed": 0},
        "recent_tests": [],
        "failed_tests": []
    }
    
    for stage_dir in log_dir.iterdir():
        if not stage_dir.is_dir():
            continue
            
        stage_name = stage_dir.name
        summary["by_stage"][stage_name] = {"passed": 0, "failed": 0}
        
        for status_dir in stage_dir.iterdir():
            if not status_dir.is_dir() or status_dir.name not in ["passed", "failed"]:
                continue
                
            status = status_dir.name
            log_files = list(status_dir.glob("*.json"))
            
            summary["by_stage"][stage_name][status] = len(log_files)
            summary["by_status"][status] += len(log_files)
            summary["total_tests"] += len(log_files)
            
            # Collect recent tests and failed tests
            for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    test_info = {
                        "file": str(log_file),
                        "stage": stage_name,
                        "status": status,
                        "algorithm": data.get("algorithm", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "duration_ms": data.get("duration_ms", 0),
                        "errors": data.get("errors", [])
                    }
                    
                    summary["recent_tests"].append(test_info)
                    
                    if status == "failed":
                        summary["failed_tests"].append(test_info)
                        
                except Exception as e:
                    print(f"Warning: Failed to read {log_file}: {e}")
    
    # Sort recent tests by timestamp
    summary["recent_tests"].sort(key=lambda x: x["timestamp"], reverse=True)
    
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print formatted summary of test logs."""
    print("=" * 60)
    print("OPENAI STAGE TEST LOG SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 OVERVIEW:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['by_status']['passed']}")
    print(f"   Failed: {summary['by_status']['failed']}")
    
    print(f"\n📁 BY STAGE:")
    for stage, counts in summary["by_stage"].items():
        total = counts["passed"] + counts["failed"]
        print(f"   {stage:12} | Total: {total:3} | Passed: {counts['passed']:3} | Failed: {counts['failed']:3}")
    
    if summary["recent_tests"]:
        print(f"\n🕒 RECENT TESTS (last 5):")
        for test in summary["recent_tests"][:5]:
            status_icon = "✅" if test["status"] == "passed" else "❌"
            duration = f"{test['duration_ms']:.0f}ms" if test["duration_ms"] else "N/A"
            print(f"   {status_icon} {test['stage']:12} | {test['algorithm']:8} | {duration:8} | {test['timestamp'][:19]}")
    
    if summary["failed_tests"]:
        print(f"\n❌ FAILED TESTS:")
        for test in summary["failed_tests"]:
            print(f"   {test['stage']:12} | {test['algorithm']:8} | {test['timestamp'][:19]}")
            for error in test["errors"][:2]:  # Show first 2 errors
                print(f"      └─ {error[:80]}...")


def show_test_details(log_file: Path) -> None:
    """Show detailed information about a specific test log."""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("=" * 60)
        print(f"TEST DETAILS: {log_file.name}")
        print("=" * 60)
        
        print(f"\n📋 BASIC INFO:")
        print(f"   Test Name: {data.get('test_name', 'N/A')}")
        print(f"   Algorithm: {data.get('algorithm', 'N/A')}")
        print(f"   Stage: {data.get('stage', 'N/A')}")
        print(f"   Status: {data.get('status', 'N/A')}")
        print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
        print(f"   Duration: {data.get('duration_ms', 0):.0f}ms")
        print(f"   Retries: {data.get('retries', 0)}")
        
        if data.get("outputs"):
            print(f"\n📤 OUTPUTS:")
            outputs = data["outputs"]
            if isinstance(outputs, dict):
                for key, value in list(outputs.items())[:5]:  # Show first 5 fields
                    if isinstance(value, dict):
                        print(f"   {key}: {list(value.keys())}")
                    else:
                        print(f"   {key}: {str(value)[:60]}...")
            else:
                print(f"   {str(outputs)[:200]}...")
        
        if data.get("errors"):
            print(f"\n❌ ERRORS:")
            for i, error in enumerate(data["errors"], 1):
                print(f"   {i}. {error}")
        
        if data.get("token_usage"):
            print(f"\n💰 TOKEN USAGE:")
            usage = data["token_usage"]
            for key, value in usage.items():
                print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="View OpenAI stage test logs")
    parser.add_argument("--log-dir", default="tests/logs", help="Log directory path")
    parser.add_argument("--show", help="Show details for specific log file")
    parser.add_argument("--stage", help="Filter by stage name")
    parser.add_argument("--status", choices=["passed", "failed"], help="Filter by status")
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return
    
    if args.show:
        show_test_details(Path(args.show))
        return
    
    summary = get_log_summary(log_dir)
    
    # Apply filters
    if args.stage:
        summary["by_stage"] = {args.stage: summary["by_stage"].get(args.stage, {"passed": 0, "failed": 0})}
    
    if args.status:
        summary["recent_tests"] = [t for t in summary["recent_tests"] if t["status"] == args.status]
        summary["failed_tests"] = [t for t in summary["failed_tests"] if t["status"] == args.status]
    
    print_summary(summary)


if __name__ == "__main__":
    main()
