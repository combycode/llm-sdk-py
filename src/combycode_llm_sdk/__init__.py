"""The public surface of the CombyCode LLM SDK.

PARTLY STUBBED. Every name the reviewed examples import exists here from the
first commit, so the surface can be checked before any behaviour backs it; the
ones not yet ported raise NotImplementedError, so nothing can pass by accident.
Names are replaced, module by module, as each is transposed. See PORTING.md.

Real today: `LLM`, `AsyncLLM`, `complete`, `acomplete`, `tool`, `Tool`,
`Agent`, `delegate`, `handoff`, `Observer`, `ClientPool`, `ReflectAndRetry`,
`TransportResponse`.

`ToolCallStartEvent` here is the HOOK event (`bus/events.py`), carrying a `ctx`.
`combycode_llm_sdk.events.ToolCallStartEvent` is the STREAM event of the same
name, carrying an id and a name off the wire. They are different events about
different things; exporting the stream one would satisfy the surface check while
meaning something else. Stream events reach callers through `LLM.stream` and
`combycode_llm_sdk.streaming.parse_stream`.
"""

from __future__ import annotations

#: The distribution's version. This literal is the ONLY place it is written:
#: `pyproject.toml` declares the version dynamic and hatchling reads it from
#: here, so the package and the wheel cannot disagree about what they are.
__version__ = "0.1.0"

from .agent.history import ConversationHistory as ConversationHistory
from .agent.history import HistoryEntry as HistoryEntry
from .agent.layers import LAYER_AGENTLOOP_CONTEXT as LAYER_AGENTLOOP_CONTEXT
from .agent.layers import LAYER_AGENTLOOP_SYSTEM as LAYER_AGENTLOOP_SYSTEM
from .agent.layers import LAYER_CONTEXT_GUARD_SUMMARY as LAYER_CONTEXT_GUARD_SUMMARY
from .agent.layers import LAYER_EXECUTOR_TOOL_EXAMPLES as LAYER_EXECUTOR_TOOL_EXAMPLES
from .agent.layers import LAYER_LAZY_TOOLS as LAYER_LAZY_TOOLS
from .agent.layers import LAYER_LEGACY_SYSTEM as LAYER_LEGACY_SYSTEM
from .agent.layers import LAYER_MEMORY as LAYER_MEMORY
from .agent.layers import PRIORITY_AGENTLOOP_CONTEXT as PRIORITY_AGENTLOOP_CONTEXT
from .agent.layers import PRIORITY_AGENTLOOP_SYSTEM as PRIORITY_AGENTLOOP_SYSTEM
from .agent.layers import PRIORITY_CHAT_FACTS as PRIORITY_CHAT_FACTS
from .agent.layers import PRIORITY_CONTEXT_GUARD_SUMMARY as PRIORITY_CONTEXT_GUARD_SUMMARY
from .agent.layers import (
    PRIORITY_EXECUTOR_TOOL_EXAMPLES as PRIORITY_EXECUTOR_TOOL_EXAMPLES,
)
from .agent.layers import PRIORITY_LAZY_TOOLS as PRIORITY_LAZY_TOOLS
from .agent.layers import PRIORITY_LEGACY_SYSTEM as PRIORITY_LEGACY_SYSTEM
from .agent.layers import PRIORITY_MEMORY as PRIORITY_MEMORY
from .agent.layers import write_agentloop_context as write_agentloop_context
from .agent.layers import write_agentloop_system as write_agentloop_system
from .agent.layers import write_lazy_tools_protocol as write_lazy_tools_protocol

# Explicit re-export form: this module IS the public surface, so a name that
# arrives here is being published, not merely imported.
from .agent.reflect_retry import ReflectAndRetry as ReflectAndRetry
from .agent.tool_key import describe_tool as describe_tool
from .agent.tool_key import tool_key as tool_key
from .approval import ApprovalDecision as ApprovalDecision
from .approval import ApprovalGate as ApprovalGate
from .approval import ApprovalRequest as ApprovalRequest
from .approval import PendingToolCall as PendingToolCall
from .approval import result_for as result_for
from .batch import AutoBatcher as AutoBatcher
from .batch import BatchStrategy as BatchStrategy
from .batch import BatchTicket as BatchTicket
from .bus.events import CompletionEvent as CompletionEvent
from .bus.events import HookEvent as HookEvent
from .bus.events import RetryContext as RetryContext
from .bus.events import RetryEvent as RetryEvent
from .bus.events import ToolCallStartEvent as ToolCallStartEvent
from .bus.events import WarningEvent as WarningEvent
from .cache import Cache as Cache
from .cache import MemoryCacheStore as MemoryCacheStore
from .cache import PersistentCacheStore as PersistentCacheStore
from .calibration import CalibrationObservation as CalibrationObservation
from .calibration import OutputCalibrationStore as OutputCalibrationStore
from .calibration import observation_from_completion as observation_from_completion
from .configuration import ConfigurationPlugin as ConfigurationPlugin
from .context.context_guard import ContextGuard as ContextGuard
from .context.facts import ExtractedFact as ExtractedFact
from .context.facts import merge_facts as merge_facts
from .context.facts import parse_facts_block as parse_facts_block
from .context.facts import render_facts_block as render_facts_block
from .context.facts import write_facts_block as write_facts_block
from .context.guard import ContextMeasurer as ContextMeasurer
from .context.guard import Measured as Measured
from .context.registry import ContextLayer as ContextLayer
from .context.registry import ContextRegistry as ContextRegistry
from .context.strategies import ANCHOR_MARKER as ANCHOR_MARKER
from .context.strategies import AnchoredStrategy as AnchoredStrategy
from .context.strategies import LayeredStrategy as LayeredStrategy
from .context.strategies import TruncateStrategy as TruncateStrategy
from .context.strategies import merge_anchor as merge_anchor
from .context.strategy_types import ContextStrategy as ContextStrategy
from .context.strategy_types import ReactContext as ReactContext
from .context.strategy_types import StrategyDecision as StrategyDecision
from .context.strategy_types import TriggerLevel as TriggerLevel
from .context.tools import ContextTools as ContextTools
from .context.tools import NoopContextTools as NoopContextTools
from .context.tools import RunnerContextTools as RunnerContextTools
from .context.tools import StrategyToolsImpl as StrategyToolsImpl
from .embeddings import EmbedResult as EmbedResult
from .embeddings import embed as embed
from .estimate import Budget as Budget
from .estimate import BudgetExceededError as BudgetExceededError
from .estimate import Estimator as Estimator
from .estimate import UnknownModelError as UnknownModelError
from .estimate import estimate_cost as estimate_cost
from .files import AnthropicFileAdapter as AnthropicFileAdapter
from .files import DefaultFileStrategy as DefaultFileStrategy
from .files import FileAttachment as FileAttachment
from .files import FilesRegistry as FilesRegistry
from .files import GoogleFileAdapter as GoogleFileAdapter
from .files import OpenAIFileAdapter as OpenAIFileAdapter
from .files import XaiFileAdapter as XaiFileAdapter
from .helpers.agent import Agent as Agent
from .helpers.batch import BatchItemResult as BatchItemResult
from .helpers.batch import BatchJob as BatchJob
from .helpers.batch import batch as batch
from .helpers.batch import submit_batch as submit_batch
from .helpers.collection import Collection as Collection
from .helpers.collection import create_collection as create_collection
from .helpers.consolidate import ConsolidateAgent as ConsolidateAgent
from .helpers.consolidate import ConsolidateAnswer as ConsolidateAnswer
from .helpers.consolidate import ConsolidateJudge as ConsolidateJudge
from .helpers.consolidate import ConsolidateResult as ConsolidateResult
from .helpers.consolidate import ConsolidateRound as ConsolidateRound
from .helpers.consolidate import aconsolidate as aconsolidate
from .helpers.consolidate import consolidate as consolidate
from .helpers.conversation_export import ConversationZip as ConversationZip
from .helpers.conversation_export import conversation_to_markdown as conversation_to_markdown
from .helpers.conversation_export import conversation_to_zip as conversation_to_zip
from .helpers.count_tokens import count_tokens as count_tokens
from .helpers.delegate import delegate as delegate
from .helpers.delegate import handoff as handoff
from .helpers.engine import Engine as Engine
from .helpers.llm import LLM as LLM
from .helpers.llm import AsyncLLM as AsyncLLM
from .helpers.mcp import McpConnection as McpConnection
from .helpers.mcp import connect_mcp as connect_mcp
from .helpers.mcp import mcp_toolset as mcp_toolset
from .helpers.models import clear_live_models_cache as clear_live_models_cache
from .helpers.models import list_models_live as list_models_live
from .helpers.observer import Observer as Observer
from .helpers.one_shot import acomplete as acomplete
from .helpers.one_shot import complete as complete
from .helpers.pool import ClientPool as ClientPool
from .helpers.realtime import Realtime as Realtime
from .helpers.route import RouteResult as RouteResult
from .helpers.route import route as route
from .helpers.select_model import filter_aliases as filter_aliases
from .helpers.select_model import filter_facets as filter_facets
from .helpers.select_model import select as select
from .helpers.select_model import select_models as select_models
from .helpers.server import create_server as create_server
from .helpers.tool import Tool as Tool
from .helpers.tool import tool as tool
from .internal_tools import InternalToolRunner as InternalToolRunner
from .internal_tools import LLMToolDefinition as LLMToolDefinition
from .internal_tools import LocalBackend as LocalBackend
from .internal_tools import ToolRegistry as ToolRegistry
from .internal_tools import define_llm_tool as define_llm_tool
from .internal_tools.builtin import BUILTIN_TOOLS as BUILTIN_TOOLS
from .internal_tools.builtin import register_builtin_tools as register_builtin_tools
from .internal_tools.ids import ParsedToolId as ParsedToolId
from .internal_tools.ids import format_tool_id as format_tool_id
from .internal_tools.ids import id_without_version as id_without_version
from .internal_tools.ids import matches_version as matches_version
from .internal_tools.ids import parse_tool_id as parse_tool_id
from .internal_tools.ids import try_parse_tool_id as try_parse_tool_id
from .llm.types.messages import AssistantPhase as AssistantPhase
from .llm.types.messages import Content as Content
from .llm.types.messages import ContentPart as ContentPart
from .llm.types.messages import DataSource as DataSource
from .llm.types.messages import Message as Message
from .llm.types.messages import MessageOrigin as MessageOrigin
from .llm.types.messages import Role as Role
from .llm.types.messages import ToolCaller as ToolCaller
from .llm.types.messages import ToolCallerType as ToolCallerType
from .llm.types.messages import content_parts as content_parts
from .llm.types.messages import content_text as content_text
from .llm.types.messages import final_answer_text as final_answer_text
from .logger import ConsoleSink as ConsoleSink
from .logger import LogEvent as LogEvent
from .logger import Logger as Logger
from .media import MediaOutput as MediaOutput
from .media import MediaResult as MediaResult
from .moderation import ModerationResult as ModerationResult
from .moderation import moderate as moderate
from .moderation import moderation_guardrail as moderation_guardrail
from .network.errors import LLMError as LLMError
from .network.retry import RetryOverride as RetryOverride
from .permissions import PermissionPolicy as PermissionPolicy
from .permissions import Rule as Rule
from .persistence import FilePersistence as FilePersistence
from .persistence import MemoryPersistence as MemoryPersistence
from .persistence import Persistence as Persistence
from .provenance import OpenAIProvenanceAdapter as OpenAIProvenanceAdapter
from .provenance import ProvenanceReport as ProvenanceReport
from .provenance import check_provenance as check_provenance
from .results import ToolCall as ToolCall
from .results import ToolResult as ToolResult
from .retrieval import Corpus as Corpus
from .retrieval import RetrievalHit as RetrievalHit
from .scheduler import ScheduledTask as ScheduledTask
from .scheduler import Scheduler as Scheduler
from .scheduler import parse_duration as parse_duration
from .server import HttpRequest as HttpRequest
from .server import HttpResponse as HttpResponse
from .server import OaiServer as OaiServer
from .server import ServerEntry as ServerEntry
from .server.oai import build_models_list as build_models_list
from .steps import ChainStepInfo as ChainStepInfo
from .steps import Step as Step
from .steps import achain as achain
from .steps import aparallel as aparallel
from .steps import chain as chain
from .steps import parallel as parallel
from .telemetry import Span as Span
from .telemetry import TelemetryAdapter as TelemetryAdapter
from .telemetry import TelemetryResource as TelemetryResource
from .telemetry import parse_traceparent as parse_traceparent
from .telemetry import to_otlp_id as to_otlp_id
from .telemetry import to_otlp_value as to_otlp_value
from .tokens import HeuristicCounter as HeuristicCounter
from .tokens import HybridCounter as HybridCounter
from .tokens import TiktokenCounter as TiktokenCounter
from .tokens import message_chars as message_chars
from .tool_catalog import ToolCatalog as ToolCatalog
from .transcription import TranscriptionResult as TranscriptionResult
from .transcription import transcribe as transcribe
from .transport import TransportResponse as TransportResponse
from .util.json_schema import validate_json_schema as validate_json_schema


