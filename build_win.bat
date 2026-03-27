@echo off
REM Build solanity-safe for Windows with CUDA
REM Requires CUDA Toolkit installed (nvcc in PATH)

where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    echo Then add nvcc to PATH ^(usually C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin^)
    exit /b 1
)

echo Building solanity-safe...
mkdir build 2>nul

cd src\cuda-ecc-ed25519

nvcc -O3 -o ..\..\build\solanity.exe ^
    vanity.cu ^
    -I.. ^
    -I..\cuda-headers ^
    -DENDIAN_NEUTRAL -DLTC_NO_ASM ^
    -lcurand ^
    --expt-relaxed-constexpr ^
    -gencode arch=compute_89,code=sm_89 ^
    -gencode arch=compute_90,code=sm_90

cd ..\..

if exist build\solanity.exe (
    echo.
    echo SUCCESS: build\solanity.exe
    echo.
    echo Usage: build\solanity.exe
    echo Edit src\config.h to change the vanity pattern
) else (
    echo.
    echo BUILD FAILED - check errors above
)
