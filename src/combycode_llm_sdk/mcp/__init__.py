"""Talking to an MCP server.

`connect_mcp` in `helpers/mcp.py` is the entry point; this package is what it is
built from -- the protocol registry, the JSON-RPC transports, the client, and
the adapter that turns a server's tools into ordinary `Tool` objects.

Three transports: stdio (a child process), Streamable HTTP (a URL), and a
WebSocket (a supported extra rather than a spec transport). OAuth 2.1 guards
authenticated servers, and the subsystems the 2026-07-28 wire adds --
`subscriptions/listen`, tasks, and `input_required` retries -- are here too.

Transposed from `unified-library-ts/src/plugins/mcp/`.
"""

from __future__ import annotations

from .client import (
    DEFAULT_CLIENT_INFO,
    DEFAULT_TASK_POLL_SECONDS,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    TERMINAL_TASK_STATUSES,
    McpClient,
    McpClientOptions,
)
from .errors import McpError, McpErrorCode
from .http import HttpTransport
from .input_required import (
    DEFAULT_INPUT_REQUIRED_MAX_ROUNDS,
    is_input_required,
    run_input_required_driver,
)
from .oauth import (
    AuthServerMetadata,
    McpAuthProvider,
    McpOAuth,
    McpOAuthClientInfo,
    McpOAuthClientMetadata,
    McpOAuthTokens,
    McpUnauthorizedError,
    finish_mcp_auth,
)
from .protocol import (
    MCP_HANDSHAKE_PROTOCOL_VERSIONS,
    MCP_KNOWN_PROTOCOL_VERSIONS,
    MCP_LATEST_HANDSHAKE_VERSION,
    MCP_LATEST_MODERN_VERSION,
    MCP_MODERN_PROTOCOL_VERSIONS,
    McpEra,
    is_handshake_mcp_version,
    is_modern_mcp_version,
    mcp_era_of,
    newest_mutual_modern_version,
)
from .result_cache import McpResultCache
from .sampling import (
    McpSamplingConfig,
    McpSamplingHandler,
    McpSamplingViaLLM,
    sampling_handler_with,
)
from .stdio import StdioTransport, safe_env
from .subscriptions import (
    MCP_SUBSCRIPTION_ID_META_KEY,
    McpServerEvent,
    McpSubscription,
    McpSubscriptionEnd,
    McpSubscriptionFilter,
    event_from_wire,
    subscription_id_from,
)
from .tools import (
    mcp_content_to_result,
    mcp_prompt_to_messages,
    mcp_tool_to_tool,
    sanitize_namespace,
)
from .transport import (
    BaseJsonRpcTransport,
    IncomingHandlers,
    McpTransport,
    OnStreamEnd,
    OnStreamOpen,
)
from .url_guard import McpSsrfError, SsrfGuardOptions, assert_safe_auth_url
from .ws import WsTransport

__all__ = [
    "DEFAULT_CLIENT_INFO",
    "DEFAULT_INPUT_REQUIRED_MAX_ROUNDS",
    "DEFAULT_TASK_POLL_SECONDS",
    "DEFAULT_TASK_TIMEOUT_SECONDS",
    "MCP_HANDSHAKE_PROTOCOL_VERSIONS",
    "MCP_KNOWN_PROTOCOL_VERSIONS",
    "MCP_LATEST_HANDSHAKE_VERSION",
    "MCP_LATEST_MODERN_VERSION",
    "MCP_MODERN_PROTOCOL_VERSIONS",
    "MCP_SUBSCRIPTION_ID_META_KEY",
    "TERMINAL_TASK_STATUSES",
    "AuthServerMetadata",
    "BaseJsonRpcTransport",
    "HttpTransport",
    "IncomingHandlers",
    "McpAuthProvider",
    "McpClient",
    "McpClientOptions",
    "McpEra",
    "McpError",
    "McpErrorCode",
    "McpOAuth",
    "McpOAuthClientInfo",
    "McpOAuthClientMetadata",
    "McpOAuthTokens",
    "McpResultCache",
    "McpSamplingConfig",
    "McpSamplingHandler",
    "McpSamplingViaLLM",
    "McpServerEvent",
    "McpSsrfError",
    "McpSubscription",
    "McpSubscriptionEnd",
    "McpSubscriptionFilter",
    "McpTransport",
    "McpUnauthorizedError",
    "OnStreamEnd",
    "OnStreamOpen",
    "SsrfGuardOptions",
    "StdioTransport",
    "WsTransport",
    "assert_safe_auth_url",
    "event_from_wire",
    "finish_mcp_auth",
    "is_handshake_mcp_version",
    "is_input_required",
    "is_modern_mcp_version",
    "mcp_content_to_result",
    "mcp_era_of",
    "mcp_prompt_to_messages",
    "mcp_tool_to_tool",
    "newest_mutual_modern_version",
    "run_input_required_driver",
    "safe_env",
    "sampling_handler_with",
    "sanitize_namespace",
    "subscription_id_from",
]
