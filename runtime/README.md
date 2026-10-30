# Runtime setup and operation

This directory is the common execution entry for local use and new clones. The supplied Windows x64 binary archive contains the Treyrin llama.cpp extension, matching CUDA 13.3 backend/libraries, ONNX Runtime, and Python 3.13 with CPU PyTorch. Language inference uses CUDA; cognitive and lifecycle operations execute on CPU in the server process.

## Start

From the repository root:

```powershell
.\Run-Llama.ps1
```

Q8_0 is the default. Missing weights/runtime are downloaded from the `v5` Release and verified against `artifacts.json`. Transport parts are reassembled without changing the model format. Interrupted part downloads are retained for retry. An existing model with a different hash is preserved and reported rather than overwritten.

The default runtime directory is `%LOCALAPPDATA%\Treyrin\runtime\v5-native-r3`; individual state and logs use `%LOCALAPPDATA%\Treyrin\state\v5`. Downloaded transport parts use `%LOCALAPPDATA%\Treyrin\downloads\v5`. Keep the individual-state directory when upgrading or relocating an installation. Use separate state directories for separate individuals, with one server per directory.

The launcher runs in the background, waits for `/health`, and prints the endpoint, PID and paths. It binds to `127.0.0.1`. The default context setting is 262144, with automatic memory fitting; actual allocation is reported in the server log. This is distinct from archived long-term source history. Output generation has no launcher-imposed fixed token cap.

Example overrides:

```powershell
.\Run-Llama.ps1 -RuntimeRoot 'D:\TreyrinRuntime' -StateDirectory 'D:\TreyrinState' -Port 18888
.\Run-Llama.ps1 -ContextSize 32768 -Threads 6
```

For CPU-only operation, use `-GpuLayers 0 -Fit off` and a context size suitable for available RAM. BF16 is opt-in through `-Variant BF16` and requires more memory. The provided release has been prepared for Windows x64; other platforms require a source build and their own dependency validation.

## Send a request

The server supplies its web interface at the printed address. A PowerShell HTTP request can use:

```powershell
$body = @{model='Treyrin-v5'; messages=@(@{role='user';content='What is 2 + 2? Reply with the number.'})} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Uri 'http://127.0.0.1:18888/v1/chat/completions' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

`-ApiKey` optionally enables the server's API-key check. Add a matching bearer header to requests when using it. Native status, decision and lifecycle operations are documented in [NATIVE-API.md](../docs/NATIVE-API.md).

## Stop and inspect

```powershell
.\runtime\Stop.ps1
# With an explicit state location:
.\runtime\Stop.ps1 -StateDirectory 'D:\TreyrinState'
```

Stopping checks the recorded executable and model before terminating the PID. State remains on disk. Startup output identifies the log directory; inspect `server.stderr.log` after an error. A runtime/model hash mismatch, missing Release asset or unsupported upstream architecture should be resolved at its reported stage.

## Source and rebuild

[source/UPSTREAM.json](source/UPSTREAM.json) pins the exact upstream commit and patch hash. [source/treyrin.patch](source/treyrin.patch) contains the loader, computation, state and server changes. [source/model/](source/model/) exposes the source embedded in the distributed model, with bundle identity in `MODEL-SOURCE.json`. Exporter scripts are supplied alongside the patch.

The binary build uses LLVM-MinGW x86_64/UCRT, CMake/Ninja, shared libraries, dynamic GGML backends and ONNX Runtime C API v31. The matching CUDA backend and redistributable libraries are included in the runtime archive. The architecture DLLs were built with `GGML_CUDA=OFF` and `GGML_BACKEND_DL=ON`, then deployed with that matching CUDA backend. A CPU build alone does not supply CUDA computation.

To prepare source:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git llama-treyrin
git -C llama-treyrin checkout 48d22e295e2b86b47366c16390794f3e05ba970a
git -C llama-treyrin apply --check (Resolve-Path '.\runtime\source\treyrin.patch')
git -C llama-treyrin apply (Resolve-Path '.\runtime\source\treyrin.patch')
```

For the tested LLVM-MinGW toolchain, configure CMake with that toolchain's C/C++ compilers, Ninja, `BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=ON`, `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_BUILD_EXAMPLES=OFF`, `LLAMA_BUILD_MTMD=OFF`, `LLAMA_CURL=OFF` and `LLAMA_OPENSSL=OFF`; build `llama-server`, `llama-cli` and `llama-quantize`. Replace the corresponding architecture/host DLLs while retaining the matching runtime dependencies. The patch supplies the ONNX headers and loads `onnxruntime.dll` dynamically. Rebuilt binaries need their own deployment checks.

The extension is pinned to its source revision. It is not a patch verified against arbitrary newer upstream revisions. An unmodified official b12000 build rejects the current `treyrin` architecture identifier.
