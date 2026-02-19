# Werkzeugkoppler

Werkzeugkoppler is an **OpenAI Chat Completions** proxy for people who live in the terminal.

It sits between **OpenAI-compatible clients** (Open-WebUI, vim-ai, your own curl aliases) and an **upstream LLM server** (llama.cpp, etc.). You configure your “stack” once (system messages, optional RAG, tools/actions) and then reuse the same setup across every client.

This is a tool for people who want leverage. If you enable command execution, you are choosing power over guardrails.

## Mental model

```
client -->  Werkzeugkoppler  -->  upstream LLM 
                |
                +-- init_messages (system prompts)
                +-- last_user_message_readers (RAG hooks via commands)
                +-- MCP servers (tool definitions)
                +-- actions (local command-backed tools)
                +-- @@direct commands (raw shell control, if enabled)
```

Both `actions` and `@@` direct commands ultimately execute local shell commands via Werkzeugkoppler.
The difference is who requests execution: `actions` are requested by the upstream model (tool-calling), while `@@` direct commands are requested explicitly by the user (typed in the client).


## What you get

- One **service URL** for all clients (`service_base_url`)
- Central **system messages** (`init_messages`)
- Optional “bring-your-own-RAG” via external commands (`last_user_message_readers`)
- Tools from **MCP servers** (`mcp_servers`)
- Local **actions** exposed as tools (model-triggered; you predefine the command) (`actions`)
- Optional **direct shell execution** via chat (user-triggered via `@@...`)

## Scope

- Supported endpoints: **only** `/v1/chat/completions` and `/v1/models`
- No embeddings/audio/images endpoints
- Streaming is supported, but Werkzeugkoppler normalizes to **one choice** (`n=1`, choice index `0`)

## Security note (mandatory, but also true)

If you enable `actions` and/or `@@` direct commands, this service can execute commands on the host.
That is the point. Do not expose it to people you do not fully trust.

Operationally:

- `actions` can be executed because the model decided to call an action.
- `@@` direct commands are executed only when the user types them.


Practical baseline:

- do not run as root
- bind to `127.0.0.1` unless you know why you are not
- set `service_api_key` when reachable over LAN
- if you want remote control, put it behind a reverse proxy and a firewall

## Requirements

- Python **3.10+**
- Tested on Linux. Other platforms are untested.

## Installation

From source:

```bash
git clone https://github.com/GhostWithAHat/werkzeugkoppler
cd werkzeugkoppler
python3 -m pip install -r requirements.txt
```

## Quickstart

### 1) Create a config

```bash
cp config.yaml.min_example config.yaml
```

### 2) Edit the minimum fields

- `upstream_base_url`: upstream LLM base URL (example: `http://127.0.0.1:10000`)
- `service_base_url`: where Werkzeugkoppler listens (example: `http://127.0.0.1:8000`)
- optional: `service_api_key` (recommended as soon as you bind to LAN)

### 3) Run

```bash
python3 -m werkzeugkoppler --config config.yaml
```

### 4) Test

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8000/v1/models
```

```bash
curl -s -H 'Content-Type: application/json' http://127.0.0.1:8000/v1/chat/completions -d '{
  "model": "whatever",
  "messages": [
    { "role": "user", "content": "Say hi in one sentence." }
  ]
}'
```

If you set `service_api_key`, add:

```bash
-H 'Authorization: Bearer <service_api_key>'
```

### 5) Point your client to `service_base_url`

Consult the readme of your client.

## Recommended: run it via systemd

A pragmatic setup for the intended audience (a box you actually control).

### Suggested layout

- Code: `/opt/werkzeugkoppler`
- Config: `/etc/werkzeugkoppler/config.yaml`
- User: `werkzeugkoppler` (dedicated, non-root)

### Example setup

```bash
sudo useradd --system --home /nonexistent --shell /usr/sbin/nologin werkzeugkoppler || true

sudo mkdir -p /opt /etc/werkzeugkoppler
sudo git clone https://github.com/GhostWithAHat/werkzeugkoppler /opt/werkzeugkoppler

sudo python3 -m venv /opt/werkzeugkoppler/venv
sudo /opt/werkzeugkoppler/venv/bin/pip install -r /opt/werkzeugkoppler/requirements.txt

sudo cp /opt/werkzeugkoppler/config.yaml.min_example /etc/werkzeugkoppler/config.yaml
sudo chown -R werkzeugkoppler:werkzeugkoppler /opt/werkzeugkoppler /etc/werkzeugkoppler
```

Edit `/etc/werkzeugkoppler/config.yaml`, then add a unit file:

`/etc/systemd/system/werkzeugkoppler.service`

```ini
[Unit]
Description=Werkzeugkoppler (OpenAI-compatible proxy)
After=network.target

[Service]
Type=simple
User=werkzeugkoppler
Group=werkzeugkoppler
WorkingDirectory=/opt/werkzeugkoppler
ExecStart=/opt/werkzeugkoppler/venv/bin/python -m werkzeugkoppler --config /etc/werkzeugkoppler/config.yaml
Restart=on-failure
RestartSec=1
Environment=PYTHONUNBUFFERED=1

# Hardening (optional)
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now werkzeugkoppler
```

Follow logs:

```bash
journalctl -u werkzeugkoppler -f
```

Config reload is automatic (no restart required), but write YAML **atomically** (write to a temp file and rename).

## Configuration reference

### Reload behavior

Werkzeugkoppler reloads the YAML without restart.
If another process writes the YAML automatically, write it **atomically**, otherwise it may read a half-written file.

### Bind address

Werkzeugkoppler binds to the host+port parsed from `service_base_url`.
There are no separate bind parameters.

### Top-level fields (overview)

- `upstream_base_url` (required): where requests are forwarded
- `upstream_api_key` (optional): bearer token sent upstream as `Authorization: Bearer <key>`
- `service_base_url` (required): where clients connect
- `service_api_key` (optional): protects `/v1/models` and `/v1/chat/completions`
- `upstream_connect_retries` (optional): retry behavior if upstream is unavailable
- `upstream_retry_interval_ms` (optional): fixed delay between retries (milliseconds)
- `init_messages` (optional): injected into every request
- `last_user_message_readers` (optional): RAG hooks via external commands
- `mcp_servers` (optional): tools from MCP servers
- `actions` (optional): local tools executed via commands
- `allowed_direct_commands` (optional): allowlist patterns for `@@` direct command execution
- `logging.level` (optional): logging verbosity

### Minimal config

```yaml
upstream_base_url: "http://127.0.0.1:10000"
service_base_url: "http://127.0.0.1:8000"

logging:
  level: "INFO"
```

## Authentication

### Upstream auth (`upstream_api_key`)

If `upstream_api_key` is set, Werkzeugkoppler sends:

- `Authorization: Bearer <upstream_api_key>`

Applied per request (GET `/v1/models`, non-stream chat, stream chat).

### Service auth (`service_api_key`)

If `service_api_key` is set, Werkzeugkoppler requires:

- `Authorization: Bearer <service_api_key>`

Applies to:
- `/v1/models`
- `/v1/chat/completions`

Does **not** apply to:
- `/healthz`

## System messages (`init_messages`)

Injected at the start of every request:

```yaml
init_messages:
  - role: "system"
    content: |
      You are a precise assistant.
      Keep answers short and factual.
```

## RAG via `last_user_message_readers` (you provide the reader)

Werkzeugkoppler does not ship a search engine. It executes whatever you configure.

### Contract

- The command must be executable on the host where Werkzeugkoppler runs.
- It must accept the last user message (typically via `$LAST_USER_MESSAGE` in arguments).
- It must print text (or JSON) to stdout.

Readers are executed **once per request** and cached in a request-local dictionary by reader name.
No conversation-level cache and no TTL.

There is no built-in truncation. Keep reader output small.

### Example

```yaml
last_user_message_readers:
  - name: "rag"
    command: "/usr/local/bin/query_embeddings"
    arguments:
      - "search"
      - "--index"
      - "/path/to/index"
      - "--results-only"
      - "--query"
      - "$LAST_USER_MESSAGE"
```

Inject into a system message:

```yaml
init_messages:
  - role: "system"
    content: |
      Additional context:
      { last_user_message_reader_output:rag }
```

## MCP servers

Register MCP servers and expose their tools:

```yaml
mcp_servers:
  - server_id: "demo_http_mcp"
    transport: "http"
    url: "http://127.0.0.1:18080/mcp/"
```

## Actions (local tool execution)

Actions are tools backed by local commands. They are exposed to the upstream model and executed when called.

Key point: the *request* to execute comes from the model (via tool-calling). The *command line* is still your configuration: the model chooses the action name and fills the declared parameters, Werkzeugkoppler runs the configured command.


```yaml
actions:
  - name: "memory_add"
    description: "Persist one memory entry"
    command: "/usr/local/bin/memories"
    arguments:
      - "add"
      - "--text"
      - "$TEXT"
    parameters:
      - name: "TEXT"
        type: "insecure_string"
        description: "Text content"
    # optional:
    # timeout: 60
```

### Timeout and output

- `timeout` default is **60 seconds**
- `stderr` is intentionally ignored
- process exit code is currently not evaluated; only stdout matters

Stdout handling:

- if stdout is valid JSON: forwarded as JSON tool output
- otherwise: forwarded as an array of lines

## Direct commands (`allowed_direct_commands`)

If a message starts with `@@`, Werkzeugkoppler parses the remainder with `shlex.split()` and checks **only the first token** against the allowlist patterns.

Key point: here the command line comes from the user. Werkzeugkoppler applies the allowlist check and then executes that command.


If allowed, the command is executed via a **shell** (pipes, redirection, etc. work).

Tight allowlist:

```yaml
allowed_direct_commands:
  - "echo"
  - "ls"
  - "find"
  - "systemctl"
```

### Recommended “remote control” mode

If you explicitly want full control from chat clients, allow everything:

```yaml
allowed_direct_commands:
  - "*"
```

This turns your chat client into a remote shell (by design).

## Upstream failure handling (`upstream_connect_retries`)

Retries are triggered on:

- `httpx.TransportError`
- `httpx.HTTPStatusError` with status **404**, **429**, or **>= 500**
- incomplete response payloads (for example “response payload is not completed”)

Values:

- `0`: no retries
- `-1`: infinite retries (streams status while retrying)
- `N > 0`: retry up to N times

Retry delay:

- `upstream_retry_interval_ms` (default: **1000 ms**), fixed delay, no exponential backoff

While retrying (and streaming), Werkzeugkoppler sends a reasoning delta status block, then switches to the upstream’s live stream as soon as it recovers.

## Streaming behavior

SSE format:

- `data: { "object": "chat.completion.chunk", ... }`
- `data: [DONE]`

Caveat: responses are normalized to **one choice** (`n=1`, index `0`).

## Troubleshooting

- 401/403:
  - `service_api_key` is set but the client sends no (or a wrong) `Authorization` header.

- Upstream not reachable:
  - verify `upstream_base_url`
  - confirm upstream is running and reachable
  - if `upstream_connect_retries` is `-1`, your client will see retry status blocks until upstream is back

- Reader/action returns nothing:
  - check paths and permissions
  - run the command manually first
  - remember: there is no truncation; keep output small

- Weird behavior after config updates:
  - your config writer is not atomic; write to a temp file and rename
