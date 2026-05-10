"""FastAPI entrypoint for the PaperDesk backend skeleton."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents import (
    LibraryRetrieverAgent,
    PaperSearchAgent,
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
    OpenAlexClient,
    PaperSearchService,
    PdfParser,
    QueryTranslationService,
    ResearchOrchestrator,
    ResearchWorkspaceService,
    TextChunker,
)
from app.vectorstores import ChromaVectorStore


@lru_cache(maxsize=1)
def get_repository() -> SQLiteRepository:
    settings = get_settings()
    return SQLiteRepository(settings.sqlite_file)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(settings.embedding_model)


@lru_cache(maxsize=1)
def get_vectorstore() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        settings.chroma_storage_path,
        get_embedding_service(),
    )


def get_library_repository():
    return get_repository().library


def get_research_repository():
    return get_repository().research


def get_paper_repository():
    return get_repository().paper


def get_report_repository():
    return get_repository().report


@lru_cache(maxsize=1)
def get_paper_search_service() -> PaperSearchService:
    settings = get_settings()
    return PaperSearchService(
        openalex_client=OpenAlexClient(
            base_url=settings.openalex_base_url,
            api_key=settings.openalex_api_key,
        ),
        arxiv_client=ArxivClient(base_url=settings.arxiv_base_url),
        translation_service=QueryTranslationService(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        ),
    )


@lru_cache(maxsize=1)
def get_document_library_service() -> DocumentLibraryService:
    settings = get_settings()
    return DocumentLibraryService(
        repository=get_library_repository(),
        vectorstore=get_vectorstore(),
        upload_dir=settings.upload_path,
        pdf_parser=PdfParser(),
        text_chunker=TextChunker(),
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
def get_research_orchestrator() -> ResearchOrchestrator:
    return ResearchOrchestrator(
        research_repository=get_research_repository(),
        paper_repository=get_paper_repository(),
        library_repository=get_library_repository(),
        report_repository=get_report_repository(),
        topic_planner=TopicPlannerAgent(),
        paper_search_agent=PaperSearchAgent(get_paper_search_service()),
        library_retriever=LibraryRetrieverAgent(get_vectorstore()),
        reading_summarizer=ReadingSummarizerAgent(),
        report_writer=ReportWriterAgent(),
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

    from app.api.routes import documents, papers, reports, research

    app.include_router(documents.router, prefix="/api")
    app.include_router(papers.router, prefix="/api")
    app.include_router(reports.router, prefix="/api")
    app.include_router(research.router, prefix="/api")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
