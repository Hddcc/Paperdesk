"""FastAPI entrypoint for the PaperDesk backend skeleton."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.domains.paper.research_agents import (
    LibraryRetrieverAgent,
    PaperAnalysisAgent,
    PaperSearchAgent,
    PaperSelectionAgent,
    ReadingSummarizerAgent,
    ReportWriterAgent,
    TopicPlannerAgent,
)
from app.config import Settings, get_settings
from app.application import ChatUseCase, PaperUploadUseCase, ReportUseCase, WorkspaceUseCase
from app.agent.tools import ToolRegistry
from app.domains.artifact import ExportService
from app.domains.paper import (
    DocumentLibraryService,
    KnowledgeIngestionService,
    PaperAnalysisService,
    PaperSearchService,
    PaperSelectionService,
    PdfParser,
    QueryTranslationService,
    RagService,
    ReportLifecycleService,
    TextChunker,
)
from app.domains.workspace import WorkbenchService, WorkspaceFileService
from app.infrastructure.files import ContextFileStore, FileAssetService, FileTextExtractor
from app.infrastructure.integrations import ArxivClient, OpenAlexClient
from app.infrastructure.llm import EmbeddingService
from app.infrastructure.vectorstore import MilvusBootstrapService
from app.repositories import SQLiteRepository
from app.services import (
    ChatMemoryService,
    ChatService,
    ContextAssembler,
    ContextBudgetService,
    ContextCompactionService,
    ResearchContextAssembler,
    ResearchWorkspaceService,
)
from app.services.research_orchestrator import ResearchOrchestrator
from app.agent.observability import AgentCoreTraceAdapter
from app.agent.runtimes.experimental import KnowledgePlannerRuntime, ReflectionRuntime
from app.agent.skills import SkillRegistry
from app.runtime import KnowledgeAgentRuntime
from app.vectorstores import MilvusVectorStore

logger = logging.getLogger(__name__)


def _warmup_vectorstore(app: FastAPI, settings: Settings) -> None:
    """Warm up Milvus in the background so the API can still start serving."""
    deadline = time.time() + max(settings.milvus_start_timeout_seconds, 10)
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            get_milvus_bootstrap_service().ensure_running()
            get_vectorstore().ensure_available()
            app.state.vectorstore_status = "ready"
            app.state.vectorstore_error = None
            return
        except Exception as exc:
            last_error = exc
            app.state.vectorstore_status = "starting"
            app.state.vectorstore_error = str(exc)
            if not _is_retryable_milvus_warmup_error(exc):
                break
            time.sleep(2)

    app.state.vectorstore_status = "failed"
    app.state.vectorstore_error = str(last_error) if last_error else "unknown Milvus warmup error"
    logger.warning(
        "Milvus warmup failed for %s: %s",
        settings.effective_milvus_uri,
        app.state.vectorstore_error,
    )


def _warmup_embedding_model(app: FastAPI, settings: Settings) -> None:
    """Warm up the embedding model in the background to avoid first-upload downloads."""
    try:
        get_embedding_service().preload()
        app.state.embedding_status = "ready"
        app.state.embedding_error = None
    except Exception as exc:
        app.state.embedding_status = "failed"
        app.state.embedding_error = str(exc)
        logger.warning("Embedding warmup failed for %s: %s", settings.embedding_model, exc)


def _is_retryable_milvus_warmup_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    retry_markers = (
        "proxy is not ready yet",
        "service unavailable",
        "channel_state=ready",
        "fail connecting to server",
        "server unavailable",
        "timed out",
    )
    return any(marker in message for marker in retry_markers)


@lru_cache(maxsize=1)
def get_repository() -> SQLiteRepository:
    settings = get_settings()
    return SQLiteRepository(settings.sqlite_file)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        settings.embedding_model,
        cache_dir=settings.embedding_cache_path,
        hf_endpoint=settings.embedding_hf_endpoint,
        local_files_only=settings.embedding_local_files_only,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> MilvusVectorStore:
    settings = get_settings()
    return MilvusVectorStore(
        uri=settings.effective_milvus_uri,
        token=settings.milvus_token,
        database=settings.milvus_database,
        collection_name=settings.milvus_collection,
        embedding_service=get_embedding_service(),
    )


@lru_cache(maxsize=1)
def get_milvus_bootstrap_service() -> MilvusBootstrapService:
    settings = get_settings()
    return MilvusBootstrapService(
        uri=settings.effective_milvus_uri,
        auto_start=settings.milvus_auto_start,
        timeout_seconds=settings.milvus_start_timeout_seconds,
        runtime_dir=settings.milvus_runtime_path,
        container_name=settings.milvus_container_name,
        image=settings.milvus_image,
        docker_desktop_executable=settings.docker_desktop_executable,
    )


def get_library_repository():
    return get_repository().library


def get_category_repository():
    return get_repository().category


def get_chunk_repository():
    return get_repository().chunk


def get_research_repository():
    return get_repository().research


def get_paper_repository():
    return get_repository().paper


def get_report_repository():
    return get_repository().report


def get_runtime_repository():
    return get_repository().runtime


def get_chat_repository():
    return get_repository().chat


def get_file_asset_repository():
    return get_repository().file_asset


def get_workspace_file_repository():
    return get_repository().workspace_file


@lru_cache(maxsize=1)
def get_report_lifecycle_service() -> ReportLifecycleService:
    return ReportLifecycleService(
        chat_repository=get_chat_repository(),
        research_repository=get_research_repository(),
        report_repository=get_report_repository(),
    )


@lru_cache(maxsize=1)
def get_workbench_service() -> WorkbenchService:
    return WorkbenchService(
        settings=get_settings(),
        library_repository=get_library_repository(),
        chat_repository=get_chat_repository(),
        file_repository=get_file_asset_repository(),
        report_repository=get_report_repository(),
        runtime_repository=get_runtime_repository(),
        knowledge_agent_runtime=get_knowledge_agent_runtime(),
    )


@lru_cache(maxsize=1)
def get_file_asset_service() -> FileAssetService:
    settings = get_settings()
    return FileAssetService(
        file_repository=get_file_asset_repository(),
        chat_repository=get_chat_repository(),
        storage_dir=settings.file_asset_path,
        max_upload_bytes=settings.file_asset_max_upload_bytes,
        text_extractor=FileTextExtractor(),
    )


@lru_cache(maxsize=1)
def get_workspace_file_service() -> WorkspaceFileService:
    settings = get_settings()
    return WorkspaceFileService(
        workspace_repository=get_workspace_file_repository(),
        chat_repository=get_chat_repository(),
        workspace_base_dir=settings.workspace_path,
        max_file_bytes=settings.workspace_file_max_bytes,
    )


@lru_cache(maxsize=1)
def get_context_file_store() -> ContextFileStore:
    return ContextFileStore(get_settings())


@lru_cache(maxsize=1)
def get_context_budget_service() -> ContextBudgetService:
    return ContextBudgetService(get_settings())


@lru_cache(maxsize=1)
def get_context_compaction_service() -> ContextCompactionService:
    settings = get_settings()
    return ContextCompactionService(
        settings,
        get_context_file_store(),
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
    )


@lru_cache(maxsize=1)
def get_context_assembler() -> ContextAssembler:
    return ContextAssembler(
        budget_service=get_context_budget_service(),
        compaction_service=get_context_compaction_service(),
        file_store=get_context_file_store(),
    )


@lru_cache(maxsize=1)
def get_research_context_assembler() -> ResearchContextAssembler:
    return ResearchContextAssembler(
        budget_service=get_context_budget_service(),
        settings=get_settings(),
    )


@lru_cache(maxsize=1)
def get_query_translation_service() -> QueryTranslationService:
    settings = get_settings()
    return QueryTranslationService(
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
    )


@lru_cache(maxsize=1)
def get_paper_search_service() -> PaperSearchService:
    settings = get_settings()
    return PaperSearchService(
        openalex_client=OpenAlexClient(
            base_url=settings.openalex_base_url,
            api_key=settings.openalex_api_key,
        ),
        arxiv_client=ArxivClient(base_url=settings.arxiv_base_url),
        translation_service=get_query_translation_service(),
    )


@lru_cache(maxsize=1)
def get_document_library_service() -> DocumentLibraryService:
    settings = get_settings()
    return DocumentLibraryService(
        repository=get_library_repository(),
        category_repository=get_category_repository(),
        vectorstore=get_vectorstore(),
        upload_dir=settings.upload_path,
        ingestion_service=get_knowledge_ingestion_service(),
    )


@lru_cache(maxsize=1)
def get_export_service() -> ExportService:
    settings = get_settings()
    return ExportService(settings.report_path)


@lru_cache(maxsize=1)
def get_research_workspace_service() -> ResearchWorkspaceService:
    settings = get_settings()
    return ResearchWorkspaceService(settings.workspace_path)


@lru_cache(maxsize=1)
def get_report_writer() -> ReportWriterAgent:
    settings = get_settings()
    return ReportWriterAgent(
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
    )


@lru_cache(maxsize=1)
def get_knowledge_ingestion_service() -> KnowledgeIngestionService:
    return KnowledgeIngestionService(
        document_repository=get_library_repository(),
        chunk_repository=get_chunk_repository(),
        vectorstore=get_vectorstore(),
        pdf_parser=PdfParser(),
        text_chunker=TextChunker(),
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        library_repository=get_library_repository(),
        chunk_repository=get_chunk_repository(),
        vectorstore=get_vectorstore(),
        translation_service=get_query_translation_service(),
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
    )


@lru_cache(maxsize=1)
def get_chat_memory_service() -> ChatMemoryService:
    return ChatMemoryService(
        chat_repository=get_chat_repository(),
        library_repository=get_library_repository(),
        file_store=get_context_file_store(),
    )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    settings = get_settings()
    return ChatService(
        chat_repository=get_chat_repository(),
        library_repository=get_library_repository(),
        file_repository=get_file_asset_repository(),
        file_asset_base_dir=settings.file_asset_path,
        category_repository=get_category_repository(),
        rag_service=get_rag_service(),
        memory_service=get_chat_memory_service(),
        context_assembler=get_context_assembler(),
        workspace_file_service=get_workspace_file_service(),
        agent_orchestrator=get_agent_core_trace_adapter(),
        knowledge_agent_runtime=get_knowledge_agent_runtime(),
        knowledge_planner_runtime=get_knowledge_planner_runtime(),
        reflection_runtime=get_reflection_runtime(),
        enable_research_from_knowledge=settings.enable_research_from_knowledge,
        enable_auto_reflection=settings.enable_auto_reflection,
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
    )


@lru_cache(maxsize=1)
def get_chat_use_case() -> ChatUseCase:
    return ChatUseCase(get_chat_service())


@lru_cache(maxsize=1)
def get_paper_upload_use_case() -> PaperUploadUseCase:
    return PaperUploadUseCase(get_document_library_service())


@lru_cache(maxsize=1)
def get_report_use_case() -> ReportUseCase:
    return ReportUseCase(
        report_repository=get_report_repository(),
        report_lifecycle_service=get_report_lifecycle_service(),
        export_service=get_export_service(),
    )


@lru_cache(maxsize=1)
def get_workspace_use_case() -> WorkspaceUseCase:
    return WorkspaceUseCase(
        workbench_service=get_workbench_service(),
        workspace_file_service=get_workspace_file_service(),
        file_asset_service=get_file_asset_service(),
    )


@lru_cache(maxsize=1)
def get_knowledge_agent_runtime() -> KnowledgeAgentRuntime:
    settings = get_settings()
    return KnowledgeAgentRuntime(
        document_library_service=get_document_library_service(),
        category_repository=get_category_repository(),
        research_repository=get_research_repository(),
        runtime_repository=get_runtime_repository(),
        rag_service=get_rag_service(),
        vectorstore=get_vectorstore(),
        file_store=get_context_file_store(),
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
        enable_subagent_execution=settings.enable_subagent_execution,
        enable_skill_context_prompt_injection=settings.enable_skill_context_prompt_injection,
        enable_skill_context_paper_qa_lightweight_only=settings.enable_skill_context_paper_qa_lightweight_only,
    )


@lru_cache(maxsize=1)
def get_agent_core_trace_adapter() -> AgentCoreTraceAdapter:
    settings = get_settings()
    return AgentCoreTraceAdapter(
        research_repository=get_research_repository(),
        runtime_repository=get_runtime_repository(),
        tool_registry=ToolRegistry(
            enable_experimental_mcp=settings.enable_experimental_mcp,
            enable_mcp_in_knowledge=settings.enable_mcp_in_knowledge,
        ),
        skill_registry=SkillRegistry(),
    )


@lru_cache(maxsize=1)
def get_knowledge_planner_runtime() -> KnowledgePlannerRuntime:
    return KnowledgePlannerRuntime(
        knowledge_agent_runtime=get_knowledge_agent_runtime(),
        runtime_repository=get_runtime_repository(),
    )


@lru_cache(maxsize=1)
def get_reflection_runtime() -> ReflectionRuntime:
    settings = get_settings()
    return ReflectionRuntime(
        knowledge_agent_runtime=get_knowledge_agent_runtime(),
        runtime_repository=get_runtime_repository(),
        memory_service=get_chat_memory_service(),
        model=settings.effective_llm_model,
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
        persist_lessons_to_memory=settings.enable_auto_reflection,
    )


@lru_cache(maxsize=1)
def get_paper_analysis_agent() -> PaperAnalysisAgent:
    return PaperAnalysisAgent(PaperAnalysisService(get_rag_service()))


@lru_cache(maxsize=1)
def get_paper_selection_agent() -> PaperSelectionAgent:
    return PaperSelectionAgent(PaperSelectionService(get_paper_search_service()))


@lru_cache(maxsize=1)
def get_research_orchestrator() -> ResearchOrchestrator:
    settings = get_settings()
    return ResearchOrchestrator(
        research_repository=get_research_repository(),
        paper_repository=get_paper_repository(),
        library_repository=get_library_repository(),
        report_repository=get_report_repository(),
        runtime_repository=get_runtime_repository(),
        topic_planner=TopicPlannerAgent(),
        paper_search_agent=PaperSearchAgent(get_paper_search_service()),
        library_retriever=LibraryRetrieverAgent(
            get_vectorstore(),
            translation_service=get_query_translation_service(),
            rag_service=get_rag_service(),
        ),
        reading_summarizer=ReadingSummarizerAgent(),
        report_writer=get_report_writer(),
        export_service=get_export_service(),
        workspace_service=get_research_workspace_service(),
        context_assembler=get_research_context_assembler(),
        enable_experimental_mcp=settings.enable_experimental_mcp,
        enable_subagent_execution=settings.enable_subagent_execution,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        get_repository()
        app.state.vectorstore_status = "starting"
        app.state.vectorstore_uri = runtime_settings.effective_milvus_uri
        app.state.vectorstore_error = None
        app.state.embedding_model = runtime_settings.embedding_model
        app.state.embedding_error = None
        if runtime_settings.embedding_warmup_on_start:
            app.state.embedding_status = "starting"
            threading.Thread(
                target=_warmup_embedding_model,
                args=(app, runtime_settings),
                daemon=True,
                name="paperdesk-embedding-warmup",
            ).start()
        else:
            app.state.embedding_status = "disabled"
        if runtime_settings.uses_embedded_milvus:
            try:
                get_vectorstore().ensure_available()
                app.state.vectorstore_status = "ready"
            except Exception as exc:
                app.state.vectorstore_status = "failed"
                app.state.vectorstore_error = str(exc)
                raise RuntimeError(
                    "Embedded Milvus failed to start. Run `uv sync` to install "
                    "`pymilvus[milvus_lite]`, or set MILVUS_URI to an external Milvus service."
                ) from exc
        else:
            threading.Thread(
                target=_warmup_vectorstore,
                args=(app, runtime_settings),
                daemon=True,
                name="paperdesk-milvus-warmup",
            ).start()
        yield

    app = FastAPI(
        title=runtime_settings.app_name,
        version=runtime_settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.get_cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.routes import chat, documents, export, papers, rag, reports, research, workbench

    app.include_router(chat.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(documents.category_router, prefix="/api")
    app.include_router(export.router, prefix="/api")
    app.include_router(papers.router, prefix="/api")
    app.include_router(rag.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")
    app.include_router(research.router, prefix="/api")
    app.include_router(workbench.router, prefix="/api")

    @app.get("/healthz")
    def healthz() -> dict[str, str | None]:
        return {
            "status": "ok",
            "vectorstore_status": getattr(app.state, "vectorstore_status", "unknown"),
            "vectorstore_uri": getattr(app.state, "vectorstore_uri", None),
            "vectorstore_error": getattr(app.state, "vectorstore_error", None),
            "embedding_status": getattr(app.state, "embedding_status", "unknown"),
            "embedding_model": getattr(app.state, "embedding_model", None),
            "embedding_error": getattr(app.state, "embedding_error", None),
        }

    return app


app = create_app()
