"""
Command-line interface for the ARDA pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .pipeline import Pipeline
from .workspace import workspace_manager
from .runtime import DefaultAgentRunner
from .observability.manager import ObservabilityManager

try:
    from .agents.openai_runner import OpenAIAgentRunner
except Exception:  # pragma: no cover - resolved at runtime when requested
    OpenAIAgentRunner = None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ARDA (Automated RTL Design with Agents): Convert Python algorithms to SystemVerilog RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create bundle from Python file
  arda --create-bundle my_algorithm.py my_bundle.txt

  # Create bundle from directory
  arda --create-bundle my_project/ project_bundle.txt

  # Run with algorithm bundle file
  arda test_algorithms/bpf16_bundle.txt

  # Run with Vivado synthesis for Xilinx FPGAs
  arda test_algorithms/bpf16_bundle.txt --synthesis-backend vivado --fpga-family xc7a100t

  # Run with open-source Yosys for iCE40 FPGAs
  arda test_algorithms/bpf16_bundle.txt --synthesis-backend yosys --fpga-family ice40hx8k

  # Auto-detect best synthesis backend
  arda test_algorithms/bpf16_bundle.txt --synthesis-backend auto

  # Save results to JSON file
  arda test_algorithms/bpf16_bundle.txt --output results.json

  # Extract generated RTL files
  arda test_algorithms/bpf16_bundle.txt --extract-rtl output_dir/

  # Legacy CLI (supported for compatibility)
  alg2sv test_algorithms/bpf16_bundle.txt
        """
    )

    parser.add_argument(
        'bundle_file',
        nargs='?',
        help='Path to algorithm bundle file'
    )

    parser.add_argument(
        '--bundle', '-b',
        help='Inline algorithm bundle string'
    )

    parser.add_argument(
        '--output', '-o',
        help='Save results to JSON file'
    )

    parser.add_argument(
        '--extract-rtl', '-e',
        help='Extract generated RTL files to directory'
    )

    parser.add_argument(
        '--create-bundle', '-c',
        nargs=2,
        metavar=('SOURCE', 'OUTPUT'),
        help='Create bundle from Python file or directory'
    )

    parser.add_argument(
        '--workspace-info', '-w',
        help='Show workspace information after run'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--synthesis-backend', '-s',
        choices=['auto', 'vivado', 'yosys', 'symbiflow'],
        default='auto',
        help='FPGA synthesis backend to use (default: auto-detect)'
    )

    parser.add_argument(
        '--fpga-family',
        help='FPGA family for synthesis (e.g., xc7a100t, ice40hx8k, ecp5)'
    )

    parser.add_argument(
        '--agent-runner',
        choices=['auto', 'default', 'openai'],
        default='auto',
        help='Select which agent runner implementation to use (default: auto)',
    )

    args = parser.parse_args()

    # Handle bundle creation
    if args.create_bundle:
        from .bundle_utils import create_bundle
        source_path, output_path = args.create_bundle
        try:
            create_bundle(source_path, output_path)
            print(f"Bundle created successfully: {output_path}")
            return
        except Exception as e:
            print(f"Error creating bundle: {e}")
            return

    # Validate arguments
    if not args.bundle_file and not args.bundle:
        parser.error("Either bundle_file or --bundle must be provided")

    if args.bundle_file and args.bundle:
        parser.error("Cannot specify both bundle_file and --bundle")

    try:
        # Load algorithm bundle
        if args.bundle_file:
            if args.verbose:
                print(f"📁 Loading bundle from file: {args.bundle_file}")
            with open(args.bundle_file, 'r') as f:
                algorithm_bundle = f.read()
        else:
            if args.verbose:
                print("📄 Using inline bundle string")
            algorithm_bundle = args.bundle

        # Run simplified pipeline
        if args.verbose:
            print("Starting ARDA pipeline...")
            print(f"Bundle length: {len(algorithm_bundle)} characters")
            print(f"Synthesis backend: {args.synthesis_backend}")
            if args.fpga_family:
                print(f"FPGA family: {args.fpga_family}")

        # Verbose trace hook
        observability = None
        if args.verbose:
            def _trace_emitter(agent_name: str, stage: str, event_type: str, payload: str):
                symbol = {
                    "stage_started": "START",
                    "stage_completed": "OK",
                    "stage_failed": "FAIL",
                    "tool_invoked": "TOOL",
                }.get(event_type, "INFO")
                try:
                    details = json.loads(payload)
                    payload_str = ", ".join(f"{k}={v}" for k, v in details.items())
                except Exception:
                    payload_str = payload
                print(f"{symbol} [{stage}] {event_type} {payload_str}")

            observability = ObservabilityManager(trace_emitter=_trace_emitter)

        # Choose agent runner implementation
        agent_runner = None
        runner_choice = args.agent_runner
        if runner_choice in ('auto', 'openai'):
            if OpenAIAgentRunner is None:
                if runner_choice == 'openai':
                    raise RuntimeError(
                        "OpenAI Agents SDK is not available. Install the `openai` package to use the openai runner."
                    )
            else:
                try:
                    agent_runner = OpenAIAgentRunner()
                    if args.verbose:
                        print("Using OpenAI Agents runtime")
                except Exception as exc:
                    if runner_choice == 'openai':
                        raise
                    if args.verbose:
                        print(f"Falling back to deterministic runner: {exc}")

        if agent_runner is None:
            agent_runner = DefaultAgentRunner()

        # Create and run pipeline
        pipeline = Pipeline(
            synthesis_backend=args.synthesis_backend,
            fpga_family=args.fpga_family,
            agent_runner=agent_runner,
            observability=observability,
        )
        
        import asyncio
        result = asyncio.run(pipeline.run(algorithm_bundle))

        # Display results
        if result['success']:
            print("Pipeline completed successfully!")
            
            # Extract summary from results
            results = result['results']
            synth_result = results.get('synth', {})
            spec_result = results.get('spec', {})
            
            algorithm_name = getattr(spec_result, 'name', 'Unknown')
            target_freq = getattr(spec_result, 'clock_mhz_target', 'N/A')
            achieved_freq = getattr(synth_result, 'fmax_mhz', 'N/A')
            
            print(f"   Algorithm: {algorithm_name}")
            print(f"   Target: {target_freq}MHz")
            print(f"   Achieved: {achieved_freq}MHz")
            
            # Extract resource usage
            lut_usage = getattr(synth_result, 'lut_usage', 'N/A')
            ff_usage = getattr(synth_result, 'ff_usage', 'N/A') 
            dsp_usage = getattr(synth_result, 'dsp_usage', 'N/A')
            print(f"   Resources: {lut_usage} LUTs, {ff_usage} FFs, {dsp_usage} DSPs")
            
            # Check verification status
            eval_result = results.get('evaluate', {})
            overall_score = getattr(eval_result, 'overall_score', 0)
            verification_status = "Passed" if overall_score >= 70 else "Failed"
            print(f"   Verification: {verification_status}")

            if args.workspace_info:
                workspace = workspace_manager.get_workspace(result['workspace_token'])
                if workspace:
                    print("\n📂 Generated Files:")
                    for file_path in workspace.list_files():
                        content = workspace.get_file(file_path)
                        print(f"   - {file_path} ({len(content) if content else 0} bytes)")

        else:
            print(f"Pipeline failed: {result['error']}")
            if 'details' in result and result['details']:
                print("Details:")
                if isinstance(result['details'], list):
                    for detail in result['details']:
                        print(f"   - {detail}")
                else:
                    print(f"   {result['details']}")

        # Save results to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {args.output}")

        # Extract RTL files
        if args.extract_rtl and result['success']:
            extract_rtl_files(result['workspace_token'], args.extract_rtl)
            print(f"RTL files extracted to: {args.extract_rtl}")

        # Exit with appropriate code
        sys.exit(0 if result['success'] else 1)

    except Exception as e:
        print(f"💥 Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def extract_rtl_files(workspace_token: str, output_dir: str):
    """
    Extract generated RTL files to a directory.

    Args:
        workspace_token: Workspace identifier
        output_dir: Output directory path
    """
    workspace = workspace_manager.get_workspace(workspace_token)
    if not workspace:
        print(f"Warning: Workspace {workspace_token} not found")
        return

    print(f"DEBUG: extract_rtl_files - workspace {workspace_token} found")
    print(f"DEBUG: extract_rtl_files - workspace has {len(workspace.list_files())} files:")
    for file_path in workspace.list_files():
        content = workspace.get_file(file_path)
        print(f"DEBUG:   {file_path} ({len(content) if content else 0} bytes)")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rtl_extensions = {'.sv', '.svh', '.v', '.vh', '.xdc', '.tcl', '.sdc', '.qsf'}
    extracted_count = 0

    for file_path in workspace.list_files():
        if any(file_path.endswith(ext) for ext in rtl_extensions):
            content = workspace.get_file(file_path)
            if content:
                # Create subdirectories if needed
                full_path = output_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, 'w') as f:
                    f.write(content)

                extracted_count += 1
                print(f"   📄 {file_path}")

    print(f"Extracted {extracted_count} RTL files")


if __name__ == '__main__':
    main()
