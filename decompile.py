#!/usr/bin/env python3
"""
dx2hlsl - Convert DXBC/DXIL/SPIR-V shaders to HLSL.

Usage:
    python decompile.py file.bin -o output.hlsl
    python decompile.py file.bin --stdout
    python decompile.py file.bin  # auto-generates file_decompiled.hlsl
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


# ============================================================================
# Configuration
# ============================================================================

# Tool paths (in tools/ subdirectory)
TOOL_DIR = Path(__file__).parent.resolve() / "tools"
DXBC2DXIL_PATH = TOOL_DIR / "dxbc2dxil.exe"
DXIL_SPIRV_PATH = TOOL_DIR / "dxil-spirv.exe"
SPIRV_CROSS_PATH = TOOL_DIR / "spirv-cross.exe"


# ============================================================================
# Enums and Dataclasses
# ============================================================================

class ILFormat(Enum):
    """Intermediate Language formats."""

    DXBC = auto()
    DXIL = auto()
    SPIRV = auto()


@dataclass(frozen=True, slots=True)
class DecompileConfig:
    """Configuration for decompilation."""

    verbose: bool = False


@dataclass(frozen=True, slots=True)
class DecompileResult:
    """Result of decompilation."""

    hlsl_code: str
    source_format: ILFormat
    source_file: Path


# ============================================================================
# Exceptions
# ============================================================================

class DecompileError(Exception):
    """Base exception for decompilation errors."""

    pass


class ToolNotFoundError(DecompileError):
    """Required tool not found."""

    def __init__(self, tool_name: str, path: Path) -> None:
        super().__init__(f"'{tool_name}' not found at: {path}")
        self.tool_name = tool_name
        self.path = path


class FormatDetectionError(DecompileError):
    """Failed to detect IL format."""

    def __init__(self, file_path: Path) -> None:
        super().__init__(f"Cannot detect IL format for: {file_path}")
        self.file_path = file_path


class ToolExecutionError(DecompileError):
    """Tool execution failed."""

    def __init__(self, tool_name: str, stderr: str) -> None:
        super().__init__(f"{tool_name} failed: {stderr}")
        self.tool_name = tool_name
        self.stderr = stderr


# ============================================================================
# Format Detection
# ============================================================================

# Magic bytes for format detection
_MAGIC_DXBC = b"DXBC"
_MAGIC_SPIRV = b"\x03\x02#\x07"  # SPIR-V magic number
_MAGIC_LLVM_BITCODE = b"BC"

# Extension to format mapping
_EXTENSION_MAP: dict[str, ILFormat] = {
    ".cso": ILFormat.DXBC,
    ".dxbc": ILFormat.DXBC,
    ".dxil": ILFormat.DXIL,
    ".spv": ILFormat.SPIRV,
}


def detect_format(file_path: Path) -> ILFormat | None:
    """
    Detect IL format from file extension and magic bytes.

    Args:
        file_path: Path to shader binary file

    Returns:
        Detected format or None if unknown
    """
    if not file_path.exists():
        return None

    # Fast path: check extension
    ext = file_path.suffix.lower()
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext]

    # Fallback: check magic bytes
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
    except OSError:
        return None

    if magic == _MAGIC_DXBC:
        return ILFormat.DXBC
    if magic == _MAGIC_SPIRV:
        return ILFormat.SPIRV
    if magic[:2] == _MAGIC_LLVM_BITCODE:
        return ILFormat.DXIL

    return None


def get_default_output_name(input_file: Path) -> str:
    """Generate default output filename."""
    return f"{input_file.stem}_decompiled.hlsl"


# ============================================================================
# Tool Validation
# ============================================================================

def validate_tool(path: Path, name: str) -> Path:
    """
    Validate tool exists.

    Args:
        path: Tool executable path
        name: Tool name for error messages

    Returns:
        Validated path

    Raises:
        ToolNotFoundError: If tool not found
    """
    if not path.exists():
        raise ToolNotFoundError(name, path)
    return path


def validate_tools_for_format(fmt: ILFormat) -> None:
    """Validate required tools exist for given format."""
    if fmt == ILFormat.DXBC:
        validate_tool(DXBC2DXIL_PATH, "dxbc2dxil")
        validate_tool(DXIL_SPIRV_PATH, "dxil-spirv")
    elif fmt == ILFormat.DXIL:
        validate_tool(DXIL_SPIRV_PATH, "dxil-spirv")
    validate_tool(SPIRV_CROSS_PATH, "spirv-cross")


# ============================================================================
# Decompilation Core
# ============================================================================

def convert_dxbc_to_dxil(
    input_file: Path,
    output_dxil: Path,
) -> None:
    """
    Convert DXBC to DXIL using dxbc2dxil.

    Args:
        input_file: Input DXBC file
        output_dxil: Output DXIL file path

    Raises:
        ToolExecutionError: If conversion fails
    """
    cmd: Sequence[str | Path] = [
        DXBC2DXIL_PATH,
        input_file,
        "/o",
        output_dxil,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        raise ToolExecutionError("dxbc2dxil", result.stderr)


def convert_il_to_spirv(
    input_file: Path,
    output_spv: Path,
) -> None:
    """
    Convert DXBC/DXIL to SPIR-V.

    Args:
        input_file: Input IL file
        output_spv: Output SPIR-V file path

    Raises:
        ToolExecutionError: If conversion fails
    """
    cmd: Sequence[str | Path] = [
        DXIL_SPIRV_PATH,
        input_file,
        "--output",
        output_spv,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        raise ToolExecutionError("dxil-spirv", result.stderr)


def convert_spirv_to_hlsl(
    input_spv: Path,
    output_hlsl: Path,
) -> None:
    """
    Convert SPIR-V to HLSL.

    Args:
        input_spv: Input SPIR-V file
        output_hlsl: Output HLSL file path

    Raises:
        ToolExecutionError: If conversion fails
    """
    cmd: Sequence[str | Path] = [
        SPIRV_CROSS_PATH,
        input_spv,
        "--hlsl",
        "--shader-model",
        "60",
        "--output",
        output_hlsl,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    if result.returncode != 0:
        raise ToolExecutionError("spirv-cross", result.stderr)


def decompile_shader(
    input_file: Path,
    output_file: Path | None,
    config: DecompileConfig,
) -> DecompileResult:
    """
    Decompile shader to HLSL.

    Args:
        input_file: Input shader binary
        output_file: Output HLSL file (None for auto-generated)
        config: Decompilation configuration

    Returns:
        Decompilation result

    Raises:
        DecompileError: If decompilation fails
    """
    # Validate input
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Detect format
    detected_format = detect_format(input_file)
    if detected_format is None:
        raise FormatDetectionError(input_file)

    # Validate tools
    validate_tools_for_format(detected_format)

    # Determine output file
    actual_output = output_file or (input_file.parent / get_default_output_name(input_file))
    actual_output.parent.mkdir(parents=True, exist_ok=True)

    # Perform decompilation
    if detected_format == ILFormat.SPIRV:
        # Direct conversion
        convert_spirv_to_hlsl(input_file, actual_output)
    elif detected_format == ILFormat.DXBC:
        # Three-step conversion: DXBC -> DXIL -> SPIR-V -> HLSL
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dxil = Path(tmpdir) / "temp.dxil"
            temp_spv = Path(tmpdir) / "temp.spv"
            convert_dxbc_to_dxil(input_file, temp_dxil)
            convert_il_to_spirv(temp_dxil, temp_spv)
            convert_spirv_to_hlsl(temp_spv, actual_output)
    else:
        # Two-step conversion via temporary SPIR-V
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_spv = Path(tmpdir) / "temp.spv"
            convert_il_to_spirv(input_file, temp_spv)
            convert_spirv_to_hlsl(temp_spv, actual_output)

    # Read result
    hlsl_code = actual_output.read_text(encoding="utf-8")

    return DecompileResult(
        hlsl_code=hlsl_code,
        source_format=detected_format,
        source_file=input_file,
    )


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="decompile",
        description="Decompile DXBC/DXIL/SPIR-V shaders to HLSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s shader.cso -o output.hlsl
  %(prog)s shader.bin --stdout
  %(prog)s shader.spv --format spirv
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input shader binary file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Output HLSL file (default: INPUT_decompiled.hlsl)",
    )
    parser.add_argument(
        "-s",
        "--stdout",
        action="store_true",
        help="Print result to stdout instead of writing to file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show file info and exit (no decompilation)",
    )

    return parser


def show_file_info(file_path: Path) -> int:
    """Show file information and exit."""
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    fmt = detect_format(file_path)
    size = file_path.stat().st_size

    print(f"File: {file_path}")
    print(f"Size: {size:,} bytes ({size / 1024:.2f} KB)")
    print(f"Format: {fmt.name if fmt else 'Unknown'}")

    if fmt == ILFormat.DXBC:
        print(f"dxbc2dxil: {'OK' if DXBC2DXIL_PATH.exists() else 'NOT FOUND'}")
        print(f"dxil-spirv: {'OK' if DXIL_SPIRV_PATH.exists() else 'NOT FOUND'}")
    elif fmt == ILFormat.DXIL:
        print(f"dxil-spirv: {'OK' if DXIL_SPIRV_PATH.exists() else 'NOT FOUND'}")
    if fmt:
        print(f"spirv-cross: {'OK' if SPIRV_CROSS_PATH.exists() else 'NOT FOUND'}")

    return 0


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --info mode
    if args.info:
        return show_file_info(args.input)

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Build configuration
    config = DecompileConfig(verbose=args.verbose)

    try:
        result = decompile_shader(args.input, args.output, config)

        if args.stdout:
            print(result.hlsl_code)
        else:
            # Determine actual output path for message
            actual_output = args.output or (
                args.input.parent / get_default_output_name(args.input)
            )
            logger.info(
                "[%s] %s -> %s (%d chars)",
                result.source_format.name,
                result.source_file.name,
                actual_output.name,
                len(result.hlsl_code),
            )

        return 0

    except ToolNotFoundError as e:
        logger.error("Tool not found: %s", e)
        return 1
    except FormatDetectionError as e:
        logger.error("Cannot detect format: %s", e)
        return 1
    except ToolExecutionError as e:
        logger.error("Decompilation failed: %s", e)
        return 1
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
