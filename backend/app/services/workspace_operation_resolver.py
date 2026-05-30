"""Deterministic workspace file intent resolution for chat requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

from app.models import ChatAttachment, ChatMessage, ChatMessageRequest


@dataclass(slots=True)
class WorkspaceFileReadIntent:
    relative_path: str | None
    reason: str
    clarification: str | None = None


@dataclass(slots=True)
class WorkspaceFileWriteNewIntent:
    relative_path: str | None
    content: str | None
    display_name: str | None
    file_kind: str | None
    source_message_id: str | None = None
    source_file_ids: list[str] | None = None
    source_document_ids: list[str] | None = None
    reason: str = "Explicit workspace write_new request handled deterministically."
    clarification: str | None = None
    is_absolute_export: bool = False


@dataclass(slots=True)
class WorkspaceFileOverwriteIntent:
    relative_path: str | None
    content: str | None
    display_name: str | None
    file_kind: str | None
    source_message_id: str | None = None
    source_file_ids: list[str] | None = None
    source_document_ids: list[str] | None = None
    reason: str = "Explicit workspace overwrite request handled through pending confirmation."
    clarification: str | None = None


@dataclass(slots=True)
class WorkspaceFilePendingResponse:
    action: str
    reason: str


@dataclass(slots=True)
class WorkspaceCommandBoundary:
    command_hint: str
    reason: str = "Local command execution is blocked by deterministic workspace boundary."


class WorkspacePathExtractor:
    """Extract path-like workspace candidates without touching the filesystem."""

    WORKSPACE_FILE_PATH_PATTERN = re.compile(
        r"`(?P<quoted>[^`\r\n]+)`"
        r"|(?P<plain>"
        r"[A-Za-z]:[\\/][^\s`\"'<>()\[\]{}，。；;：！!?]+"
        r"|\\\\[^\s`\"'<>()\[\]{}，。；;：！!?]+"
        r"|/(?!/)[^\s`\"'<>()\[\]{}，。；;：！!?]+"
        r"|(?:\.\./)+[^\s`\"'<>()\[\]{}，。；;：！!?]+"
        r"|\./[^\s`\"'<>()\[\]{}，。；;：！!?]+"
        r"|\.[A-Za-z0-9_.-]+(?:[\\/][^\s`\"'<>()\[\]{}，。；;：！!?]+)*"
        r"|[A-Za-z0-9_\-\u4e00-\u9fff.]+[\\/][^\s`\"'<>()\[\]{}，。；;：！!?]*"
        r"|[A-Za-z0-9_\-\u4e00-\u9fff.]+\.[A-Za-z0-9]{1,12}"
        r")",
        re.IGNORECASE,
    )

    @classmethod
    def extract_paths(cls, content: str) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for match in cls.WORKSPACE_FILE_PATH_PATTERN.finditer(cls._path_scan_text(content)):
            raw_path = (match.group("quoted") or match.group("plain") or "").strip()
            normalized = cls.normalize_candidate(raw_path)
            if normalized is None:
                continue
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
        return paths

    @staticmethod
    def instruction_segment(content: str) -> str:
        match = re.search(
            r"(\u5185\u5bb9\u5982\u4e0b|\u5185\u5bb9\u662f|\u5199\u5165\u5185\u5bb9\u662f|with\s+content|content\s+is|content\s*:)",
            content,
            flags=re.IGNORECASE,
        )
        if match is None:
            return content
        return content[: match.start()]

    @staticmethod
    def without_paths(content: str, paths: list[str]) -> str:
        cleaned = content
        variants: set[str] = set()
        for path in paths:
            variants.add(path)
            variants.add(f"./{path}")
            variants.add(path.replace("/", "\\"))
            variants.add(f"`{path}`")
        for variant in sorted(variants, key=len, reverse=True):
            cleaned = cleaned.replace(variant, " ")
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _path_scan_text(content: str) -> str:
        extension = r"(?:md|txt|json|csv|html|py|go|js|ts|vue|css|java|cpp|c|rs|ya?ml|toml)"
        return re.sub(
            rf"(\.{extension})\s*(?:、|，|,|\s+和\s+|\s+and\s+)\s*(?=(?:\./)?[A-Za-z0-9_\-\u4e00-\u9fff.]+[\\/])",
            r"\1 ",
            content,
            flags=re.IGNORECASE,
        )

    @classmethod
    def normalize_candidate(cls, raw_path: str) -> str | None:
        candidate = str(raw_path or "").strip().strip("\"'“”‘’")
        if not candidate:
            return None
        while candidate and candidate[-1] in "，。；;：！!?":
            candidate = candidate[:-1].strip()
        if candidate.endswith(".") and Path(candidate[:-1]).suffix:
            candidate = candidate[:-1]
        while candidate and candidate[-1] in ",，。；：、":
            candidate = candidate[:-1].strip()
        if not candidate or candidate.casefold().startswith(("http://", "https://")):
            return None
        if candidate.startswith("/") and not Path(candidate).suffix:
            return None
        if not cls.looks_like_workspace_path_candidate(candidate):
            return None
        normalized = candidate.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.strip() or None

    @staticmethod
    def looks_like_workspace_path_candidate(candidate: str) -> bool:
        if re.match(r"^[A-Za-z]:[\\/]", candidate):
            return True
        if candidate.startswith(("\\\\", "/", "../", "./", ".")):
            return True
        if "/" in candidate or "\\" in candidate:
            return True
        return bool(re.search(r"\.[A-Za-z0-9]{1,12}$", candidate))


class WorkspaceBoundaryGuard:
    """Keep workspace-local file requests inside the supported phase boundary."""

    @staticmethod
    def contains_any(content: str, markers: tuple[str, ...]) -> bool:
        normalized = content.casefold()
        return any(marker.casefold() in normalized for marker in markers)

    @classmethod
    def has_internal_write_boundary_intent(cls, content: str) -> bool:
        if cls.is_internal_write_concept_question(content):
            return False
        return cls.contains_any(
            content,
            (
                "write to vectorstore",
                "save to vectorstore",
                "write vectorstore",
                "import to library",
                "add to library",
                "save to library",
                "write to vector index",
                "upsert vectorstore",
                "generate chunks",
                "generate chunk",
                "write chunks",
                "save to report paper_ids",
                "write report paper_ids",
                "save to paper_ids",
                "write paper_ids",
                "assign tag",
                "add tag",
                "apply tag",
                "write tag",
                "paper_ids",
                "chunks",
                "加入论文库",
                "导入论文库",
                "保存到论文库",
                "打标签",
                "加标签",
                "应用标签",
                "写入标签",
                "向量库",
                "向量索引",
                "写入向量库",
                "写入 vectorstore",
                "写入向量索引",
                "upsert 向量库",
                "生成 chunks",
                "生成 chunk",
                "保存到 report paper_ids",
                "写入 report paper_ids",
                "保存到 paper_ids",
                "写入 paper_ids",
            ),
        )

    @classmethod
    def is_internal_write_concept_question(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "what is",
                "what are",
                "explain",
                "define",
                "meaning of",
                "什么是",
                "是什么",
                "解释",
                "概念",
            ),
        )

    @classmethod
    def has_library_boundary_marker(cls, content: str) -> bool:
        return cls.has_internal_write_boundary_intent(content)

    @classmethod
    def has_write_new_marker(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "create",
                "generate",
                "save as",
                "write a new file",
                "output to",
                "export to",
                "save locally",
                "save to local",
                "save on my computer",
                "新建",
                "创建",
                "生成",
                "保存为",
                "保存成",
                "写一个",
                "写入一个新文件",
                "另存为",
                "输出到",
                "存成",
                "保存到本地",
                "保存到电脑",
                "存到本地",
                "导出到",
            ),
        )

    @classmethod
    def has_write_boundary_marker(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "overwrite",
                "edit",
                "modify",
                "update",
                "delete",
                "remove",
                "rename",
                "move",
                "append",
                "run",
                "execute",
                "覆盖",
                "编辑",
                "修改",
                "改短",
                "删除",
                "重命名",
                "移动",
                "追加",
                "运行",
                "执行",
            ),
        )

    @classmethod
    def has_read_marker(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "read",
                "open",
                "inspect",
                "view",
                "summarize",
                "analyze",
                "explain",
                "based on",
                "workspace file",
                "读取",
                "打开",
                "查看",
                "看看",
                "总结",
                "概括",
                "分析",
                "解释",
                "根据",
                "这个文件",
                "workspace 文件",
                "工作区文件",
            ),
        )

    @classmethod
    def has_mutation_intent(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "write",
                "edit",
                "delete",
                "remove",
                "overwrite",
                "rename",
                "move",
                "save",
                "append",
                "create",
                "run",
                "execute",
                "import to library",
                "add to library",
                "save to library",
                "vectorstore",
                "vector index",
                "写入",
                "编辑",
                "修改",
                "改短",
                "删除",
                "移除",
                "覆盖",
                "重命名",
                "移动",
                "保存",
                "追加",
                "创建",
                "运行",
                "执行",
                "加入论文库",
                "导入论文库",
                "保存到论文库",
                "写入论文库",
                "加到论文库",
                "打标签",
                "向量库",
                "向量索引",
            ),
        )

    @classmethod
    def has_overwrite_marker(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "overwrite",
                "replace",
                "save over",
                "replace with",
                "覆盖",
                "替换",
                "覆盖到",
                "保存到已有文件",
            ),
        )

    @classmethod
    def has_abstract_edit_marker(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "modify",
                "edit",
                "update",
                "append",
                "add",
                "change to",
                "修改",
                "编辑",
                "更新",
                "改短一点",
                "改长一点",
                "加一段",
                "添加",
                "追加",
                "改成",
                "改为",
            ),
        )

    @classmethod
    def has_unsupported_overwrite_marker(cls, content: str) -> bool:
        return cls.contains_any(
            content,
            (
                "delete",
                "remove",
                "rename",
                "move",
                "run",
                "execute",
                "npm run",
                "pytest",
                "go test",
                "删除",
                "重命名",
                "移动",
                "运行",
                "执行",
            ),
        )


class WorkspaceIntentResolver:
    """Resolve chat text into deterministic workspace operations."""

    WORKSPACE_FILE_WRITE_NEW_EXTENSIONS = {
        ".txt": ("txt", "text/plain"),
        ".md": ("md", "text/markdown"),
        ".json": ("json", "application/json"),
        ".csv": ("csv", "text/csv"),
        ".html": ("html", "text/html"),
        ".py": ("py", "text/x-python"),
        ".go": ("go", "text/x-go"),
        ".js": ("js", "text/javascript"),
        ".ts": ("ts", "text/typescript"),
        ".vue": ("vue", "text/x-vue"),
        ".css": ("css", "text/css"),
        ".java": ("java", "text/x-java-source"),
        ".cpp": ("cpp", "text/x-c++src"),
        ".c": ("c", "text/x-csrc"),
        ".rs": ("rs", "text/rust"),
        ".yaml": ("yaml", "application/yaml"),
        ".yml": ("yaml", "application/yaml"),
        ".toml": ("toml", "application/toml"),
    }
    WORKSPACE_FILE_CONFIRM_MARKERS = (
        "confirm",
        "yes",
        "execute",
        "continue",
        "ok",
        "确认",
        "执行",
        "继续",
        "是的",
        "同意",
        "可以",
    )
    WORKSPACE_FILE_CANCEL_MARKERS = (
        "cancel",
        "no",
        "stop",
        "取消",
        "不用",
        "撤销",
        "先不",
    )
    COMMAND_HINT_MAX_CHARS = 120
    COMMAND_EXECUTION_PATTERNS = (
        r"npm\s+run\s+\S+",
        r"npm\s+test\b",
        r"npm\s+install\b",
        r"yarn\s+(?:build|test|install)\b",
        r"pnpm\s+(?:build|test|install)\b",
        r"pytest(?:\s+[\w./\\:-]+)*",
        r"python\s+-m\s+pytest(?:\s+[\w./\\:-]+)*",
        r"python\s+[\w./\\:-]+\.py",
        r"go\s+test(?:\s+[\w./\\:-]+)*",
        r"go\s+run\s+[\w./\\:-]+",
        r"go\s+build(?:\s+[\w./\\:-]+)*",
        r"node\s+[\w./\\:-]+",
        r"cargo\s+(?:test|build)\b",
        r"mvn\s+test\b",
        r"gradle\s+test\b",
        r"make(?:\s+\w+)?\b",
        r"docker\s+build\b",
        r"docker\s+compose\s+up\b",
        r"git\s+(?:status|diff|add|commit|push)\b",
        r"rm\s+-rf\b",
        r"del(?:\s+[\w./\\:-]+)?\b",
        r"copy\s+[\w./\\:-]+",
        r"move\s+[\w./\\:-]+",
        r"powershell(?:\.exe)?\b",
        r"bash\b",
        r"cmd(?:\.exe)?\b",
        r"uvicorn\s+[\w.:/-]+",
        r"pip\s+install\b",
    )
    COMMAND_EXECUTION_NATURAL_MARKERS = (
        "帮我运行",
        "帮我执行",
        "执行",
        "运行",
        "跑一下",
        "跑下",
        "启动服务",
        "运行测试",
        "运行构建",
        "打开终端执行",
        "在本地执行",
        "run ",
        "execute ",
        "start server",
        "run tests",
        "run build",
    )
    COMMAND_CONCEPT_QUESTION_MARKERS = (
        "什么是",
        "是什么",
        "做什么",
        "怎么用",
        "如何",
        "怎么",
        "区别",
        "报错",
        "一般怎么看",
        "解释",
        "含义",
        "what is",
        "what does",
        "how to",
        "how do",
        "difference",
        "explain",
        "meaning",
        "usage",
    )

    def __init__(
        self,
        *,
        pending_reader: Callable[[str], dict | None] | None = None,
        active_write_intent: Callable[[str], bool] | None = None,
        read_then_write_request: Callable[[str], bool] | None = None,
    ) -> None:
        self.pending_reader = pending_reader
        self.active_write_intent = active_write_intent or (lambda _content: False)
        self.read_then_write_request = read_then_write_request or (lambda _content: False)

    def detect_write_new_intent(
        self,
        *,
        request: ChatMessageRequest,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        history: list[ChatMessage],
    ) -> WorkspaceFileWriteNewIntent | None:
        content = request.content.strip()
        instruction_content = WorkspacePathExtractor.instruction_segment(content)
        paths = WorkspacePathExtractor.extract_paths(instruction_content)
        marker_source = WorkspacePathExtractor.without_paths(instruction_content, paths)
        has_write_new_marker = WorkspaceBoundaryGuard.has_write_new_marker(marker_source)
        has_write_boundary_marker = WorkspaceBoundaryGuard.has_write_boundary_marker(marker_source)
        has_library_boundary_marker = WorkspaceBoundaryGuard.has_library_boundary_marker(marker_source)
        if not paths:
            if self._looks_like_local_save_without_path(content):
                return WorkspaceFileWriteNewIntent(
                    relative_path=None,
                    content=None,
                    display_name=None,
                    file_kind=None,
                    reason="Local save requested without a concrete destination path.",
                    clarification=(
                        "请提供要保存到的完整本地路径，例如 `D:\\文献\\review.md` 或 `D:\\文献\\review.txt`。"
                        "也可以点击回答下方的“保存为文件”按钮，在系统保存窗口里选择位置。"
                    ),
                )
            if WorkspaceBoundaryGuard.has_internal_write_boundary_intent(content):
                return WorkspaceFileWriteNewIntent(
                    relative_path=None,
                    content=None,
                    display_name=None,
                    file_kind=None,
                    reason="Internal library, vector, report, chunk, or tag write intent is outside write_new.",
                    clarification=workspace_internal_write_boundary_message(),
                )
            return None
        if not (has_write_new_marker or has_write_boundary_marker or has_library_boundary_marker):
            return None

        if selected_document_ids or request.selected_document_ids or request.selected_file_ids or attachments:
            return WorkspaceFileWriteNewIntent(
                relative_path=None,
                content=None,
                display_name=None,
                file_kind=None,
                reason="Workspace write_new mixed with selected files, selected papers, or attachments.",
                clarification=(
                    "当前 workspace 新建文件请求需要单独处理。请取消已选择的文件、论文或附件后重新发送保存请求。"
                ),
            )
        if request.command in {"tag", "library"} or has_library_boundary_marker:
            return WorkspaceFileWriteNewIntent(
                relative_path=None,
                content=None,
                display_name=None,
                file_kind=None,
                reason="Workspace file request attempted to cross into library/vector/report writes.",
                clarification=(
                    "Workspace 文件当前只支持在本地 workspace 中新建和读取；"
                    "加入论文库、标签、向量索引或 report paper_ids 需要通过论文库功能处理。"
                ),
            )
        if len(paths) > 1:
            return WorkspaceFileWriteNewIntent(
                relative_path=None,
                content=None,
                display_name=None,
                file_kind=None,
                reason="Multiple workspace paths require clarification before write_new.",
                clarification="当前一次只支持新建一个 workspace 文件。请保留一个目标相对路径后重新发送。",
            )
        if has_write_boundary_marker:
            return WorkspaceFileWriteNewIntent(
                relative_path=paths[0],
                content=None,
                display_name=None,
                file_kind=None,
                reason="Workspace overwrite/edit/delete/rename/move/run intent is outside write_new.",
                clarification=workspace_file_write_new_boundary_message(),
            )
        if not has_write_new_marker:
            return None

        relative_path = paths[0]
        extension_info = self.WORKSPACE_FILE_WRITE_NEW_EXTENSIONS.get(Path(relative_path).suffix.casefold())
        if extension_info is None:
            return WorkspaceFileWriteNewIntent(
                relative_path=relative_path,
                content=None,
                display_name=Path(relative_path).name,
                file_kind=None,
                reason="Unsupported workspace write_new extension.",
                clarification=unsupported_workspace_write_extension_message(self.WORKSPACE_FILE_WRITE_NEW_EXTENSIONS),
            )

        explicit_content = extract_explicit_workspace_file_content(content)
        if explicit_content is not None:
            return WorkspaceFileWriteNewIntent(
                relative_path=relative_path,
                content=explicit_content,
                display_name=Path(relative_path).name,
                file_kind=extension_info[0],
                reason="Explicit workspace write_new content supplied by the user.",
                is_absolute_export=self._should_export_to_absolute_path(relative_path, marker_source),
            )

        if can_save_previous_assistant_content(content):
            previous = latest_assistant_message(history)
            if previous is not None and previous.content.strip():
                return WorkspaceFileWriteNewIntent(
                    relative_path=relative_path,
                    content=previous.content,
                    display_name=Path(relative_path).name,
                    file_kind=extension_info[0],
                    source_message_id=previous.id,
                    source_file_ids=list(previous.used_file_ids),
                    source_document_ids=list(previous.used_document_ids),
                    reason="Previous assistant message saved through deterministic workspace write_new.",
                    is_absolute_export=self._should_export_to_absolute_path(relative_path, marker_source),
                )
            return WorkspaceFileWriteNewIntent(
                relative_path=relative_path,
                content=None,
                display_name=Path(relative_path).name,
                file_kind=extension_info[0],
                reason="Workspace write_new requested previous assistant content but none exists.",
                clarification="没有找到可保存的上一条 assistant 内容。请先让助手生成内容，或在本句中明确提供要写入的内容。",
            )

        return WorkspaceFileWriteNewIntent(
            relative_path=relative_path,
            content=None,
            display_name=Path(relative_path).name,
            file_kind=extension_info[0],
            reason="Workspace write_new request has no deterministic content source.",
            clarification="请提供要写入文件的内容，或者先让助手生成内容后再说保存为该 workspace 文件。",
            is_absolute_export=self._should_export_to_absolute_path(relative_path, marker_source),
        )

    @staticmethod
    def _is_absolute_export_path(path: str) -> bool:
        return bool(re.match(r"^[A-Za-z]:/", path)) or (path.startswith("/") and not path.startswith("//"))

    @classmethod
    def _should_export_to_absolute_path(cls, path: str, marker_source: str) -> bool:
        if not cls._is_absolute_export_path(path):
            return False
        normalized = marker_source.casefold()
        markers = (
            "save to",
            "save as",
            "save into",
            "export to",
            "output to",
            "save locally",
            "save to local",
            "save on my computer",
            "保存到",
            "保存为",
            "保存成",
            "另存为",
            "存到本地",
            "保存到本地",
            "保存到电脑",
            "导出到",
            "输出到",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _looks_like_local_save_without_path(content: str) -> bool:
        normalized = content.casefold()
        markers = (
            "save locally",
            "save to local",
            "save on my computer",
            "保存到本地",
            "保存到电脑",
            "存到本地",
            "导出到本地",
        )
        return any(marker in normalized for marker in markers)

    def detect_overwrite_intent(
        self,
        *,
        request: ChatMessageRequest,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
        history: list[ChatMessage],
    ) -> WorkspaceFileOverwriteIntent | None:
        content = request.content.strip()
        instruction_content = WorkspacePathExtractor.instruction_segment(content)
        paths = WorkspacePathExtractor.extract_paths(instruction_content)
        marker_source = WorkspacePathExtractor.without_paths(instruction_content, paths)
        has_overwrite_marker = WorkspaceBoundaryGuard.has_overwrite_marker(marker_source)
        has_abstract_edit_marker = WorkspaceBoundaryGuard.has_abstract_edit_marker(marker_source)
        has_unsupported_marker = WorkspaceBoundaryGuard.has_unsupported_overwrite_marker(marker_source)
        has_library_boundary_marker = WorkspaceBoundaryGuard.has_library_boundary_marker(marker_source)
        if not paths:
            return None
        if not (has_overwrite_marker or has_abstract_edit_marker or has_unsupported_marker or has_library_boundary_marker):
            return None

        if selected_document_ids or request.selected_document_ids or request.selected_file_ids or attachments:
            return WorkspaceFileOverwriteIntent(
                relative_path=None,
                content=None,
                display_name=None,
                file_kind=None,
                reason="Workspace overwrite mixed with selected files, selected papers, or attachments.",
                clarification=(
                    "Workspace overwrite/edit requests must be handled alone. "
                    "Please clear selected files, selected papers, and attachments before retrying."
                ),
            )
        if request.command in {"tag", "library"} or has_library_boundary_marker:
            return WorkspaceFileOverwriteIntent(
                relative_path=None,
                content=None,
                display_name=None,
                file_kind=None,
                reason="Workspace overwrite request attempted to cross into library/vector/report writes.",
                clarification=(
                    "Workspace files cannot be added to the paper library, tagged, chunked, written to vectorstore, "
                    "or saved into report paper_ids in this phase."
                ),
            )
        if len(paths) > 1:
            return WorkspaceFileOverwriteIntent(
                relative_path=None,
                content=None,
                display_name=None,
                file_kind=None,
                reason="Multiple workspace paths require clarification before overwrite.",
                clarification="Only one workspace file can be edited or overwritten at a time. Please keep one target path.",
            )
        if has_unsupported_marker:
            return WorkspaceFileOverwriteIntent(
                relative_path=paths[0],
                content=None,
                display_name=None,
                file_kind=None,
                reason="Workspace delete/rename/move/run intent is outside overwrite confirmation.",
                clarification=workspace_file_overwrite_boundary_message(),
            )

        relative_path = paths[0]
        extension_info = self.WORKSPACE_FILE_WRITE_NEW_EXTENSIONS.get(Path(relative_path).suffix.casefold())
        if extension_info is None:
            return WorkspaceFileOverwriteIntent(
                relative_path=relative_path,
                content=None,
                display_name=Path(relative_path).name,
                file_kind=None,
                reason="Unsupported workspace overwrite extension.",
                clarification=unsupported_workspace_write_extension_message(self.WORKSPACE_FILE_WRITE_NEW_EXTENSIONS),
            )

        explicit_content = extract_explicit_workspace_file_overwrite_content(content)
        if explicit_content is not None and has_overwrite_marker:
            return WorkspaceFileOverwriteIntent(
                relative_path=relative_path,
                content=explicit_content,
                display_name=Path(relative_path).name,
                file_kind=extension_info[0],
                reason="Explicit workspace overwrite content supplied by the user.",
            )

        if can_overwrite_with_previous_assistant_content(content):
            previous = latest_assistant_message(history)
            if previous is not None and previous.content.strip():
                return WorkspaceFileOverwriteIntent(
                    relative_path=relative_path,
                    content=previous.content,
                    display_name=Path(relative_path).name,
                    file_kind=extension_info[0],
                    source_message_id=previous.id,
                    source_file_ids=list(previous.used_file_ids),
                    source_document_ids=list(previous.used_document_ids),
                    reason="Previous assistant message prepared for workspace overwrite confirmation.",
                )
            return WorkspaceFileOverwriteIntent(
                relative_path=relative_path,
                content=None,
                display_name=Path(relative_path).name,
                file_kind=extension_info[0],
                reason="Workspace overwrite requested previous assistant content but none exists.",
                clarification="No previous assistant content is available for overwrite. Please provide the complete new content.",
            )

        return WorkspaceFileOverwriteIntent(
            relative_path=relative_path,
            content=None,
            display_name=Path(relative_path).name,
            file_kind=extension_info[0],
            reason="Workspace overwrite/edit request has no deterministic content source.",
            clarification=(
                "Current file editing needs complete replacement content, or a request to overwrite with the previous "
                "assistant answer. Abstract edits such as shortening, refactoring, appending, or changing one field are "
                "deferred to a later edit-planning phase."
            ),
        )

    def detect_read_intent(
        self,
        *,
        request: ChatMessageRequest,
        selected_document_ids: list[str],
        attachments: list[ChatAttachment],
    ) -> WorkspaceFileReadIntent | None:
        content = request.content.strip()
        paths = WorkspacePathExtractor.extract_paths(content)
        if not paths:
            return None

        has_read_marker = WorkspaceBoundaryGuard.has_read_marker(content)
        has_mutation_marker = (
            self.active_write_intent(content)
            or self.read_then_write_request(content)
            or WorkspaceBoundaryGuard.has_mutation_intent(content)
        )
        if not has_read_marker and not has_mutation_marker:
            return None

        if selected_document_ids or request.selected_document_ids or request.selected_file_ids:
            return WorkspaceFileReadIntent(
                relative_path=None,
                reason="Workspace path mixed with selected files or papers requires clarification.",
                clarification=(
                    "当前请求同时包含 workspace 文件路径和已选择的附件/论文。为了避免混淆，请分开提问："
                    "要么读取 workspace 文件，要么分析已选择的附件/论文。"
                ),
            )
        if attachments:
            return WorkspaceFileReadIntent(
                relative_path=None,
                reason="Workspace path mixed with attachments requires clarification.",
                clarification=(
                    "当前请求同时包含 workspace 文件路径和附件。请分开提问："
                    "要么读取 workspace 文件，要么分析附件。"
                ),
            )
        if request.command in {"tag", "library"}:
            return WorkspaceFileReadIntent(
                relative_path=None,
                reason="Workspace path mixed with a library/tag command requires clarification.",
                clarification="当前 workspace 文件读取只支持只读问答。请去掉标签或论文库命令后重新提问。",
            )

        if len(paths) > 1:
            return WorkspaceFileReadIntent(
                relative_path=None,
                reason="Multiple workspace file paths require clarification before deterministic read.",
                clarification="当前一次只支持读取一个 workspace 文件。请保留一个相对路径后重新发送。",
            )
        if has_mutation_marker:
            return WorkspaceFileReadIntent(
                relative_path=None,
                reason="Workspace path mutation or command intent is outside the read-only phase.",
                clarification=(
                    "当前 workspace 文件能力只支持只读查看、总结、分析和解释。"
                    "请去掉写入、编辑、删除、重命名、移动、运行或入库要求后重新提问。"
                ),
            )
        return WorkspaceFileReadIntent(
            relative_path=paths[0],
            reason="Explicit workspace relative path read answered through read-only DirectAnswer context.",
        )

    def detect_pending_response(self, session_id: str, content: str) -> WorkspaceFilePendingResponse | None:
        if self.pending_reader is None:
            return None
        pending = self.pending_reader(session_id)
        if pending is None:
            return None
        normalized = content.casefold().strip()
        if WorkspaceBoundaryGuard.contains_any(normalized, self.WORKSPACE_FILE_CANCEL_MARKERS):
            return WorkspaceFilePendingResponse(
                action="cancel",
                reason="Workspace overwrite pending action cancelled by user.",
            )
        if WorkspaceBoundaryGuard.contains_any(normalized, self.WORKSPACE_FILE_CONFIRM_MARKERS):
            return WorkspaceFilePendingResponse(
                action="confirm",
                reason="Workspace overwrite pending action confirmed by user.",
            )
        return None

    @classmethod
    def detect_command_boundary(cls, content: str) -> WorkspaceCommandBoundary | None:
        cleaned = re.sub(r"\s+", " ", str(content or "").strip())
        if not cleaned:
            return None
        if cls.is_command_concept_question(cleaned):
            return None
        command_hint = cls.extract_command_execution_hint(cleaned)
        if command_hint is None:
            return None
        return WorkspaceCommandBoundary(command_hint=command_hint)

    @classmethod
    def extract_command_execution_hint(cls, content: str) -> str | None:
        normalized = content.casefold()
        for pattern in cls.COMMAND_EXECUTION_PATTERNS:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match is None:
                continue
            if cls.is_bare_command_request(normalized) or cls.has_command_execution_request_marker(
                normalized,
                require_command_context=False,
            ):
                return cls.safe_command_hint(match.group(0))
        if cls.has_command_execution_request_marker(normalized, require_command_context=True):
            return cls.safe_command_hint(content)
        return None

    @classmethod
    def is_bare_command_request(cls, content: str) -> bool:
        return any(
            re.fullmatch(rf"\s*{pattern}\s*", content, flags=re.IGNORECASE)
            for pattern in cls.COMMAND_EXECUTION_PATTERNS
        )

    @classmethod
    def has_command_execution_request_marker(
        cls,
        content: str,
        *,
        require_command_context: bool,
    ) -> bool:
        if not WorkspaceBoundaryGuard.contains_any(content, cls.COMMAND_EXECUTION_NATURAL_MARKERS):
            return False
        if not require_command_context:
            return True
        return WorkspaceBoundaryGuard.contains_any(
            content,
            (
                "命令",
                "终端",
                "本地",
                "测试",
                "构建",
                "服务",
                "server",
                "tests",
                "build",
                "terminal",
                "shell",
                "command",
            ),
        )

    @classmethod
    def is_command_concept_question(cls, content: str) -> bool:
        return WorkspaceBoundaryGuard.contains_any(content, cls.COMMAND_CONCEPT_QUESTION_MARKERS)

    @classmethod
    def safe_command_hint(cls, content: str) -> str:
        hint = re.sub(r"\s+", " ", str(content or "").strip())
        hint = re.sub(
            r"(?i)(api[_-]?key|token|secret|password|passwd|credential)(\s*[=:]\s*)\S+",
            r"\1\2[redacted]",
            hint,
        )
        if len(hint) > cls.COMMAND_HINT_MAX_CHARS:
            return hint[: cls.COMMAND_HINT_MAX_CHARS] + "..."
        return hint


def strip_wrapping_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    match = re.fullmatch(r"```[A-Za-z0-9_-]*\s*\n(?P<body>.*)\n```", stripped, flags=re.DOTALL)
    if match is None:
        return stripped
    return match.group("body").strip()


def extract_explicit_workspace_file_content(content: str) -> str | None:
    marker_pattern = (
        r"(?:内容如下|内容是|写入内容是|with\s+content|content\s+is|content\s*:)\s*[:：]?\s*(?P<body>.+)\s*$"
    )
    match = re.search(marker_pattern, content, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    body = strip_wrapping_code_fence(match.group("body").strip())
    return body if body.strip() else None


def extract_explicit_workspace_file_overwrite_content(content: str) -> str | None:
    patterns = (
        r"(?:content\s+is|content\s*:|with\s+content|内容如下|内容是|替换为|改成|改为)\s*[:：]?\s*(?P<body>.+)\s*$",
        r"\b(?:replace|overwrite|save\s+over)\b.+?\bwith\b\s*(?P<body>.+)\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            continue
        body = strip_wrapping_code_fence(match.group("body").strip())
        if body.strip():
            return body
    return None


def can_save_previous_assistant_content(content: str) -> bool:
    return WorkspaceBoundaryGuard.contains_any(
        content,
        (
            "save as",
            "save to",
            "save into",
            "export to",
            "output to",
            "previous",
            "last answer",
            "above",
            "刚才",
            "上面",
            "上一条",
            "保存为",
            "保存成",
            "存成",
            "另存为",
            "输出到",
        ),
    )


def can_overwrite_with_previous_assistant_content(content: str) -> bool:
    return WorkspaceBoundaryGuard.contains_any(
        content,
        (
            "previous assistant",
            "last answer",
            "above content",
            "above answer",
            "use the above",
            "use previous",
            "previous",
            "above",
            "用上面",
            "用刚才",
            "刚才的回答",
            "上面的内容",
            "上一条",
        ),
    )


def latest_assistant_message(history: list[ChatMessage]) -> ChatMessage | None:
    for message in reversed(history):
        if message.role == "assistant" and message.content.strip():
            return message
    return None


def unsupported_workspace_write_extension_message(extensions: dict[str, tuple[str, str]]) -> str:
    supported = ", ".join(sorted(extensions))
    return f"当前只支持新建安全文本/代码文件，支持的扩展名包括：{supported}。"


def workspace_file_write_new_boundary_message() -> str:
    return (
        "当前 workspace 文件能力只支持新建安全文件，不支持覆盖、编辑、删除、重命名、移动或运行脚本。"
        "覆盖和编辑需要后续确认流程。"
    )


def workspace_internal_write_boundary_message() -> str:
    return (
        "当前 workspace 文件能力只支持在本地 workspace 中新建和读取安全文件，"
        "写入向量库、生成 chunks、写入 report paper_ids、加入论文库或打标签等操作尚未开放。"
    )


def workspace_file_overwrite_boundary_message() -> str:
    return (
        "Current workspace editing only supports generating an overwrite diff for one existing text file, "
        "then applying it after confirmation. Delete, rename, move, and command execution are not open."
    )


def workspace_command_boundary_message() -> str:
    return (
        "当前 workspace 能力尚未开放命令执行。我不能运行 npm、pytest、go test、shell、git、docker "
        "等本地命令；如果需要，后续需要单独实现带沙箱和确认机制的命令执行能力。"
    )
