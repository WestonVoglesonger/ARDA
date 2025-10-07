"""
Command-line interface for ALG2SV pipeline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .pipeline import run_pipeline_sync, load_bundle_from_file
from .workspace import workspace_manager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ALG2SV: Convert Python algorithms to SystemVerilog RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with algorithm bundle file
  alg2sv test_algorithms/bpf16_bundle.txt

  # Run with Vivado synthesis for Xilinx FPGAs
  alg2sv test_algorithms/bpf16_bundle.txt --synthesis-backend vivado --fpga-family xc7a100t

  # Run with open-source Yosys for iCE40 FPGAs
  alg2sv test_algorithms/bpf16_bundle.txt --synthesis-backend yosys --fpga-family ice40hx8k

  # Auto-detect best synthesis backend
  alg2sv test_algorithms/bpf16_bundle.txt --synthesis-backend auto

  # Save results to JSON file
  alg2sv test_algorithms/bpf16_bundle.txt --output results.json

  # Extract generated RTL files
  alg2sv test_algorithms/bpf16_bundle.txt --extract-rtl output_dir/
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

    args = parser.parse_args()

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
            algorithm_bundle = load_bundle_from_file(args.bundle_file)
        else:
            if args.verbose:
                print("📄 Using inline bundle string")
            algorithm_bundle = args.bundle

        # Run pipeline
        if args.verbose:
            print("🚀 Starting ALG2SV pipeline...")
            print(f"Bundle length: {len(algorithm_bundle)} characters")
            print(f"Synthesis backend: {args.synthesis_backend}")
            if args.fpga_family:
                print(f"FPGA family: {args.fpga_family}")

        result = run_pipeline_sync(
            algorithm_bundle,
            synthesis_backend=args.synthesis_backend,
            fpga_family=args.fpga_family
        )

        # Display results
        if result['success']:
            print("✅ Pipeline completed successfully!")
            summary = result['summary']
            print(f"   Algorithm: {summary['algorithm']}")
            print(f"   Target: {summary['target_frequency']}MHz")
            print(f"   Achieved: {summary['achieved_frequency']:.1f}MHz")
            print(f"   Resources: {summary['resource_usage']['lut']} LUTs, "
                  f"{summary['resource_usage']['ff']} FFs, "
                  f"{summary['resource_usage']['dsp']} DSPs")
            print(f"   Verification: {'✅ Passed' if summary['verification_passed'] else '❌ Failed'}")

            if args.workspace_info:
                workspace = workspace_manager.get_workspace(result['workspace_token'])
                if workspace:
                    print("\n📂 Generated Files:")
                    for file_path in workspace.list_files():
                        content = workspace.get_file(file_path)
                        print(f"   - {file_path} ({len(content) if content else 0} bytes)")

        else:
            print(f"❌ Pipeline failed: {result['error']}")
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
            print(f"💾 Results saved to: {args.output}")

        # Extract RTL files
        if args.extract_rtl and result['success']:
            extract_rtl_files(result['workspace_token'], args.extract_rtl)
            print(f"📤 RTL files extracted to: {args.extract_rtl}")

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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rtl_extensions = {'.sv', '.svh', '.v', '.vh'}
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
