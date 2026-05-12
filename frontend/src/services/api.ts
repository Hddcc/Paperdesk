import type {
  ChatContextState,
  ChatMessageRequest,
  ChatSendResponse,
  ChatSession,
  ChatSessionCreateRequest,
  ChatSessionDetail,
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
  ResearchRunDetail,
  ResearchRequest,
  ResearchStreamEvent
} from "../types/models";

const baseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
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
  const response = await fetch(`${baseUrl}/api/export/${reportId}`);
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Export failed: ${response.status}`);
  }
  return response.text();
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
