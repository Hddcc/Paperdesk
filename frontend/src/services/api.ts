import type {
  LibraryDocument,
  ReportListItem,
  ResearchReport,
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
      return new Error(`${fallbackMessage}，后端连接可能已中断或正在重启。`);
    }
    return err;
  }
  return new Error(fallbackMessage);
}

export async function listDocuments(): Promise<LibraryDocument[]> {
  try {
    const response = await fetch(`${baseUrl}/api/documents`);
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

export async function listReports(): Promise<ReportListItem[]> {
  const response = await fetch(`${baseUrl}/api/reports`);
  return parseJson<ReportListItem[]>(response);
}

export async function getReport(reportId: string): Promise<ResearchReport> {
  const response = await fetch(`${baseUrl}/api/reports/${reportId}`);
  return parseJson<ResearchReport>(response);
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
