import type {
  ChatContextState,
  ChatMessageRequest,
  ChatSendResponse,
  ChatSession,
  ChatSessionCreateRequest,
  ChatSessionDetail,
  ChatStreamEvent,
  DocumentCategory,
  DocumentCategoryAssignmentRequest,
  DocumentCategoryCreateRequest,
  DocumentCategoryUpdateRequest,
  LibraryDocument,
  MemorySnapshot,
  PaperAnalysisRequest,
  PaperAnalysisResponse,
  PaperCurationRequest,
  PaperCurationResponse,
  RagAnswer,
  RagAskRequest,
  ReportListItem,
  ResearchReport,
  ResearchRuntimeState,
  ResearchRunDetail,
  ResearchRequest,
  ResearchStreamEvent,
  WorkbenchCapabilitiesResponse,
  WorkbenchConfigResponse,
  WorkbenchFileAsset,
  WorkbenchFileContextResponse,
  WorkbenchMessageTraceSummary,
  WorkspaceFile,
  WorkspaceFileSaveRequest
} from "../types/models";

const baseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function parseErrorDetail(response: Response): Promise<string> {
  const rawDetail = await response.text().catch(() => "");
  if (!rawDetail) {
    return `Request failed: ${response.status}`;
  }
  try {
    const parsed = JSON.parse(rawDetail) as { detail?: unknown };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return parsed.detail;
    }
  } catch {
    // Keep the server body when it is plain text.
  }
  return rawDetail;
}

function normalizeFetchError(err: unknown, fallbackMessage: string): Error {
  if (err instanceof Error) {
    if (err instanceof TypeError) {
      return new Error(`${fallbackMessage}，连接暂时中断，请稍后重试。`);
    }
    return err;
  }
  return new Error(fallbackMessage);
}

export async function listDocuments(): Promise<LibraryDocument[]> {
  try {
    const response = await fetch(`${baseUrl}/api/documents`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" }
    });
    return parseJson<LibraryDocument[]>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载文档失败");
  }
}

export async function uploadDocument(file: File): Promise<LibraryDocument> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch(`${baseUrl}/api/documents/upload`, {
      method: "POST",
      body: formData
    });
    return parseJson<LibraryDocument>(response);
  } catch (err) {
    throw normalizeFetchError(err, "上传文档失败");
  }
}

export async function uploadWorkbenchSessionFile(
  sessionId: string,
  file: File
): Promise<WorkbenchFileAsset> {
  const formData = new FormData();
  formData.append("file", file);
  try {
    const response = await fetch(`${baseUrl}/api/workbench/sessions/${sessionId}/files/upload`, {
      method: "POST",
      body: formData
    });
    return parseJson<WorkbenchFileAsset>(response);
  } catch (err) {
    throw normalizeFetchError(err, "上传会话文件失败");
  }
}

export async function deleteDocument(documentId: string): Promise<LibraryDocument> {
  try {
    const response = await fetch(`${baseUrl}/api/documents/${documentId}`, {
      method: "DELETE"
    });
    return parseJson<LibraryDocument>(response);
  } catch (err) {
    throw normalizeFetchError(err, "删除文档失败");
  }
}

export function getDocumentFileUrl(documentId: string): string {
  return `${baseUrl}/api/documents/${encodeURIComponent(documentId)}/file`;
}

export async function listDocumentCategories(): Promise<DocumentCategory[]> {
  try {
    const response = await fetch(`${baseUrl}/api/document-categories`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" }
    });
    return parseJson<DocumentCategory[]>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载分类失败");
  }
}

export async function createDocumentCategory(
  payload: DocumentCategoryCreateRequest
): Promise<DocumentCategory> {
  try {
    const response = await fetch(`${baseUrl}/api/document-categories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<DocumentCategory>(response);
  } catch (err) {
    throw normalizeFetchError(err, "创建分类失败");
  }
}

export async function updateDocumentCategory(
  categoryId: string,
  payload: DocumentCategoryUpdateRequest
): Promise<DocumentCategory> {
  try {
    const response = await fetch(`${baseUrl}/api/document-categories/${categoryId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<DocumentCategory>(response);
  } catch (err) {
    throw normalizeFetchError(err, "更新分类失败");
  }
}

export async function deleteDocumentCategory(categoryId: string): Promise<DocumentCategory> {
  try {
    const response = await fetch(`${baseUrl}/api/document-categories/${categoryId}`, {
      method: "DELETE"
    });
    return parseJson<DocumentCategory>(response);
  } catch (err) {
    throw normalizeFetchError(err, "删除分类失败");
  }
}

export async function assignDocumentCategories(
  documentId: string,
  payload: DocumentCategoryAssignmentRequest
): Promise<LibraryDocument> {
  try {
    const response = await fetch(`${baseUrl}/api/documents/${documentId}/categories`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<LibraryDocument>(response);
  } catch (err) {
    throw normalizeFetchError(err, "保存论文分类失败");
  }
}

export async function getWorkbenchConfig(): Promise<WorkbenchConfigResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/workbench/config`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" }
    });
    return parseJson<WorkbenchConfigResponse>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载助手配置失败");
  }
}

export async function getWorkbenchCapabilities(): Promise<WorkbenchCapabilitiesResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/workbench/capabilities`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" }
    });
    return parseJson<WorkbenchCapabilitiesResponse>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载助手能力失败");
  }
}

export async function getWorkbenchSessionFiles(
  sessionId: string
): Promise<WorkbenchFileContextResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/workbench/sessions/${sessionId}/files`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" }
    });
    return parseJson<WorkbenchFileContextResponse>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载论文库与文件状态失败");
  }
}

export async function getWorkbenchMessageTrace(
  messageId: string
): Promise<WorkbenchMessageTraceSummary> {
  try {
    const response = await fetch(`${baseUrl}/api/workbench/messages/${messageId}/trace`, {
      cache: "no-store",
      headers: { "Cache-Control": "no-cache" }
    });
    return parseJson<WorkbenchMessageTraceSummary>(response);
  } catch (err) {
    throw normalizeFetchError(err, "鍔犺浇 Trace 鎽樿澶辫触");
  }
}

export async function saveMessageAsWorkspaceFile(
  sessionId: string,
  messageId: string,
  payload: WorkspaceFileSaveRequest
): Promise<WorkspaceFile> {
  try {
    const response = await fetch(
      `${baseUrl}/api/workbench/sessions/${sessionId}/messages/${messageId}/workspace-files`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }
    );
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    return (await response.json()) as WorkspaceFile;
  } catch (err) {
    throw normalizeFetchError(err, "保存文件失败");
  }
}

export async function exportMessageToLocalPath(
  sessionId: string,
  messageId: string,
  path: string
): Promise<WorkspaceFile> {
  try {
    const response = await fetch(
      `${baseUrl}/api/workbench/sessions/${sessionId}/messages/${messageId}/export-to-path`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path })
      }
    );
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    return (await response.json()) as WorkspaceFile;
  } catch (err) {
    throw normalizeFetchError(err, "保存到本地路径失败");
  }
}

export function getWorkspaceFileDownloadUrl(sessionId: string, fileId: string): string {
  return `${baseUrl}/api/workbench/sessions/${encodeURIComponent(sessionId)}/workspace-files/${encodeURIComponent(fileId)}/download`;
}

export async function askKnowledgeQuestion(payload: RagAskRequest): Promise<RagAnswer> {
  try {
    const response = await fetch(`${baseUrl}/api/rag/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<RagAnswer>(response);
  } catch (err) {
    throw normalizeFetchError(err, "知识问答失败");
  }
}

export async function listChatSessions(): Promise<ChatSession[]> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions`);
    return parseJson<ChatSession[]>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载会话失败");
  }
}

export async function createChatSession(payload: ChatSessionCreateRequest = {}): Promise<ChatSession> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<ChatSession>(response);
  } catch (err) {
    throw normalizeFetchError(err, "创建会话失败");
  }
}

export async function deleteChatSession(sessionId: string): Promise<ChatSession> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}`, {
      method: "DELETE"
    });
    return parseJson<ChatSession>(response);
  } catch (err) {
    throw normalizeFetchError(err, "删除会话失败");
  }
}

export async function getChatSessionDetail(sessionId: string): Promise<ChatSessionDetail> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}`);
    return parseJson<ChatSessionDetail>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载会话详情失败");
  }
}

export async function getChatMemorySnapshot(sessionId: string): Promise<MemorySnapshot> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}/memory`);
    return parseJson<MemorySnapshot>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载会话记忆失败");
  }
}

export async function getChatContextState(sessionId: string): Promise<ChatContextState> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}/context`);
    return parseJson<ChatContextState>(response);
  } catch (err) {
    throw normalizeFetchError(err, "加载上下文状态失败");
  }
}

export async function sendChatMessage(
  sessionId: string,
  payload: ChatMessageRequest
): Promise<ChatSendResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<ChatSendResponse>(response);
  } catch (err) {
    throw normalizeFetchError(err, "发送消息失败");
  }
}

export async function streamChatMessage(
  sessionId: string,
  payload: ChatMessageRequest,
  onEvent: (event: ChatStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  try {
    const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}/messages/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream"
      },
      body: JSON.stringify(payload),
      signal
    });

    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(detail || `Chat stream failed: ${response.status}`);
    }

    if (!response.body) {
      throw new Error("The browser does not support stream reading.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const rawEvent = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        const event = parseSseEvent<ChatStreamEvent>(rawEvent);
        if (event) {
          onEvent(event);
          if (event.type === "error") {
            throw new Error(event.message);
          }
          if (event.type === "done") {
            return;
          }
        }
        boundary = buffer.indexOf("\n\n");
      }

      if (done) {
        break;
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    throw normalizeFetchError(err, "发送消息失败");
  }
}

export async function saveChatMessageAsReport(
  sessionId: string,
  messageId: string
): Promise<ResearchReport> {
  try {
    const response = await fetch(`${baseUrl}/api/reports/from-message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message_id: messageId
      })
    });
    return parseJson<ResearchReport>(response);
  } catch (err) {
    throw normalizeFetchError(err, "保存报告失败");
  }
}

export async function analyzePapers(payload: PaperAnalysisRequest): Promise<PaperAnalysisResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/papers/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<PaperAnalysisResponse>(response);
  } catch (err) {
    throw normalizeFetchError(err, "论文分析失败");
  }
}

export async function curatePapers(payload: PaperCurationRequest): Promise<PaperCurationResponse> {
  try {
    const response = await fetch(`${baseUrl}/api/papers/curate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    return parseJson<PaperCurationResponse>(response);
  } catch (err) {
    throw normalizeFetchError(err, "候选筛选失败");
  }
}

export async function listReports(): Promise<ReportListItem[]> {
  const response = await fetch(`${baseUrl}/api/reports`);
  return parseJson<ReportListItem[]>(response);
}

export async function getReport(reportId: string): Promise<ResearchReport> {
  const response = await fetch(`${baseUrl}/api/reports/${reportId}`);
  return parseJson<ResearchReport>(response);
}

export async function deleteReport(reportId: string): Promise<ResearchReport> {
  const response = await fetch(`${baseUrl}/api/reports/${reportId}`, {
    method: "DELETE"
  });
  return parseJson<ResearchReport>(response);
}

export async function getResearchRun(taskId: string): Promise<ResearchRunDetail> {
  const response = await fetch(`${baseUrl}/api/research/${taskId}`);
  return parseJson<ResearchRunDetail>(response);
}

export async function exportReportMarkdown(reportId: string): Promise<string> {
  const response = await fetch(`${baseUrl}/api/reports/${reportId}/export.md`);
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Export failed: ${response.status}`);
  }
  return response.text();
}

function parseSseEvent<T>(rawEvent: string): T | null {
  if (!rawEvent) {
    return null;
  }
  const dataLines: string[] = [];
  for (const line of rawEvent.split(/\r?\n/)) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  const data = dataLines.join("\n").trim();
  return data ? (JSON.parse(data) as T) : null;
}

export async function runResearchStream(
  payload: ResearchRequest,
  onEvent: (event: ResearchStreamEvent) => void
): Promise<void> {
  const response = await fetch(`${baseUrl}/api/research/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Research stream failed: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("The browser does not support stream reading.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (rawEvent.startsWith("data:")) {
        const payloadText = rawEvent.slice(5).trim();
        if (payloadText) {
          const event = JSON.parse(payloadText) as ResearchStreamEvent;
          onEvent(event);
          if (event.type === "done" || event.type === "error") {
            return;
          }
        }
      }
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }
}

export async function resumeResearchStream(
  runId: string,
  onEvent: (event: ResearchStreamEvent) => void
): Promise<void> {
  const response = await fetch(`${baseUrl}/api/research/${runId}/resume/stream`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream"
    }
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Research resume stream failed: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("The browser does not support stream reading.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      if (rawEvent.startsWith("data:")) {
        const payloadText = rawEvent.slice(5).trim();
        if (payloadText) {
          const event = JSON.parse(payloadText) as ResearchStreamEvent;
          onEvent(event);
          if (event.type === "done" || event.type === "error") {
            return;
          }
        }
      }
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }
}
