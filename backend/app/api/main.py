"""FastAPI entrypoint for the PaperDesk backend skeleton."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents import (
    LibraryRetrieverAgent,
    PaperAnalysisAgent,
    PaperSearchAgent,
    PaperSelectionAgent,
    ReadingSummarizerAgent,
    ReportWriterAgent,
    TopicPlannerAgent,
)
from app.config import Settings, get_settings
from app.repositories import SQLiteRepository
from app.services import (
    ArxivClient,
    DocumentLibraryService,
    EmbeddingService,
    ExportService,
    KnowledgeIngestionService,
    OpenAlexClient,
    PaperAnalysisService,
    PaperSearchService,
    PaperSelectionService,
    PdfParser,
    QueryTranslationService,
    RagService,
    ResearchWorkspaceService,
    TextChunker,
)
from app.services.research_orchestrator import ResearchOrchestrator
from app.vectorstores import MilvusVectorStore


@lru_cache(maxsize=1)
def get_repository() -> SQLiteRepository:
    settings = get_settings()
    return SQLiteRepository(settings.sqlite_file)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(settings.embedding_model)


@lru_cache(maxsize=1)
def get_vectorstore() -> MilvusVectorStore:
    settings = get_settings()
    return MilvusVectorStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        database=settings.milvus_database,
        collection_name=settings.milvus_collection,
        embedding_service=get_embedding_service(),
    )


def get_library_repository():
    return get_repository().library


def get_chunk_repository():
    return get_repository().chunk


def get_research_repository():
    return get_repository().research


def get_paper_repository():
    return get_repository().paper


def get_report_repository():
    return get_repository().report


@lru_cache(maxsize=1)
def get_query_translation_service() -> QueryTranslationService:
    settings = get_settings()
    return QueryTranslationService(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
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
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
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
        vectorstore=get_vectorstore(),
        translation_service=get_query_translation_service(),
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


@lru_cache(maxsize=1)
def get_paper_analysis_agent() -> PaperAnalysisAgent:
    return PaperAnalysisAgent(PaperAnalysisService(get_rag_service()))


@lru_cache(maxsize=1)
def get_paper_selection_agent() -> PaperSelectionAgent:
    return PaperSelectionAgent(PaperSelectionService(get_paper_search_service()))


@lru_cache(maxsize=1)
def get_research_orchestrator() -> ResearchOrchestrator:
    return ResearchOrchestrator(
        research_repository=get_research_repository(),
        paper_repository=get_paper_repository(),
        library_repository=get_library_repository(),
        report_repository=get_report_repository(),
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
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    app = FastAPI(title=runtime_settings.app_name, version=runtime_settings.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.get_cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.routes import documents, export, papers, rag, reports, research

    app.include_router(documents.router, prefix="/api")
    app.include_router(export.router, prefix="/api")
    app.include_router(papers.router, prefix="/api")
    app.include_router(rag.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")
    app.include_router(research.router, prefix="/api")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
