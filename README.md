# dx2hlsl

A shader decompiler that converts DXBC/DXIL/SPIR-V binaries to HLSL source code.

## Features

- **DXBC** (DirectX Bytecode) → HLSL
- **DXIL** (DirectX Intermediate Language) → HLSL
- **SPIR-V** → HLSL

## Directory Structure

```
.
├── decompile.py          # Main entry point
├── tools/                # Toolchain
│   ├── dxbc2dxil.exe     # DXBC → DXIL
│   ├── dxil-spirv.exe    # DXIL → SPIR-V
│   ├── spirv-cross.exe   # SPIR-V → HLSL
│   ├── dxilconv.dll      # DXIL conversion library
│   └── dxil-spirv-c-shared.dll
└── README.md
```

## Usage

### Basic Usage

```bash
# Auto-detect format and decompile
python decompile.py shader.cso

# Specify output file
python decompile.py shader.cso -o output.hlsl

# Output to stdout
python decompile.py shader.cso --stdout

# View file information
python decompile.py shader.cso --info
```

## Conversion Pipeline

### DXBC Files
```
DXBC → dxbc2dxil → DXIL → dxil-spirv → SPIR-V → spirv-cross → HLSL
```

### DXIL Files
```
DXIL → dxil-spirv → SPIR-V → spirv-cross → HLSL
```

### SPIR-V Files
```
SPIR-V → spirv-cross → HLSL
```

## Requirements

- Python 3.10+
- Windows (toolchain consists of Windows executables)

## Examples

```bash
$ python decompile.py sample.cso --info
File: sample.cso
Size: 30,992 bytes (30.27 KB)
Format: DXBC
dxbc2dxil: OK
dxil-spirv: OK
spirv-cross: OK

$ python decompile.py sample.cso -o output.hlsl
INFO: [DXBC] sample.cso -> output.hlsl (48342 chars)
```

## License

This project uses the following tools:
- [DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler) (MIT)
- [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) (Apache-2.0)
- [dxil-spirv](https://github.com/HansKristian-Work/dxil-spirv) (MIT)
