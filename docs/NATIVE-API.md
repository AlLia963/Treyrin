# Native model operations

The patched server exposes model operations in addition to its chat endpoint. These use the same loaded model and individual state. If an API key was configured, send its bearer header with each request.

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Server readiness |
| `/v1/models` | GET | Loaded model identity |
| `/v1/chat/completions` | POST | Language conversation with native cognitive participation |
| `/v1/treyrin/status` | GET | Lifecycle state counters, scheduler configuration and pending-action counts |
| `/v1/treyrin/memory` | GET | The current memory-read result |
| `/v1/treyrin/memory/continue` | POST | Continue the current source-memory read |
| `/v1/treyrin/control` | POST | Explicit model lifecycle operation |

For example, inspect lifecycle status:

```powershell
Invoke-RestMethod 'http://127.0.0.1:18888/v1/treyrin/status'
```

Read a closed candidate choice without applying a learning update:

```powershell
$body = @{operation='decide';question='Which number equals 2 + 2?';candidates=@('3','4')} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri 'http://127.0.0.1:18888/v1/treyrin/control' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

The control schema and transitions are defined in [native_lifecycle.py](../runtime/source/model/native_lifecycle.py). It implements status, actions, decision readout, candidate queries, Spark proposal/revision, typed record intake, intent intake, feedback learning and scheduler configuration. Operations that update experience, hypotheses, goals or adapters affect the current individual and persist their corresponding state. `request_id` receipts support idempotency at the native control layer.

Host actions remain the caller's execution responsibility. Readouts and candidate-normalized scores retain the meanings described in [ARCHITECTURE.md](ARCHITECTURE.md). The HTTP wrapper, cancellation and state-save behavior are included in [treyrin.patch](../runtime/source/treyrin.patch).
