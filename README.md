# werkzeugkoppler

## Purpose
`werkzeugkoppler` provides one centralized, reusable AI environment (prompts +
tools) for multiple clients.

The goal is to configure behavior, context, and tool access once, then expose
it through an OpenAI-compatible API endpoint that different AI clients can use.
This avoids duplicating setup in every individual client.

It also allows direct integration of your favorite command line tools,
eliminating the need for complicated MCP setups.

## How It Works
You run `werkzeugkoppler` as a local service. It:

- examines which tools your MCP servers offer,
- receives OpenAI-compatible chat requests from your client,
- injects 
  - your configured prompts,
  - tools offered by your mcp servers,
  - command line tools, that you want to offer your LLM,
- forwards to your upstream LLM,
- calls mcp tools or command line tools on demand,
- and returns real-time responses in OpenAI-compatible format.

## Installation
From the project directory:

```bash
python3 -m pip install -r requirements.txt
```

## Configuration
1. Copy the example config:

```bash
cp config.yaml.min_example config.yaml
```

2. Edit `config.yaml`:

- set your service endpoint (`service_base_url`),
- set upstream endpoint (`upstream_base_url`),
- optionally set API keys,
- configure `first_messages`, MCP servers, and/or local `actions`.

Configuration files:

- `config.yaml.min_example`: minimal setup for quick start.
- `config.yaml.full_example`: full reference with advanced options (for power
  users).

## Start
Start the gateway with the helper script:

```bash
./werkzeugkoppler.sh
```

Optional: start with a custom config path:

```bash
./werkzeugkoppler.sh /path/to/config.yaml
```

## Connect Your AI Client
In your AI client, add a new OpenAI-compatible endpoint pointing to
`service_base_url` from `config.yaml`.

Typical values:

- Base URL: `http://127.0.0.1:10001`
- API key:
  - if `service_api_key` is set: use that key,
  - if `service_api_key` is empty/unset: any non-empty placeholder may be
    accepted by some clients, or leave empty if the client allows it.

The client should use the standard OpenAI routes:

- `GET /v1/models`
- `POST /v1/chat/completions`

