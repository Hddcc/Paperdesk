import { defineStore } from "pinia";
import { computed, ref, watch } from "vue";

import {
  createChatSession,
  deleteChatSession,
  getChatSessionDetail,
  listChatSessions,
  saveChatMessageAsReport,
  streamChatMessage
} from "../services/api";
import type {
  ChatAttachment,
  ChatContextState,
  ChatMessage,
  ChatSendResponse,
  ChatStreamEvent,
  ChatSession,
  LibraryDocument,
  MemorySnapshot
} from "../types/models";

const STORAGE_KEY = "paperdesk-knowledge-draft-state";

interface PersistedKnowledgeState {
  currentSessionId?: string;
  composerText?: string;
  selectedDocumentIds?: string[];
  uploadedTaskDocumentIds?: string[];
  draftAttachments?: ChatAttachment[];
}

function readPersistedState(): PersistedKnowledgeState {
  if (typeof window === "undefined") {
    return {};
  }
  try {
    const rawValue = window.localStorage.getItem(STORAGE_KEY);
    if (!rawValue) {
      return {};
    }
    const parsed = JSON.parse(rawValue) as PersistedKnowledgeState;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function persistableAttachments(attachments: ChatAttachment[]) {
  return attachments
    .filter((attachment) => attachment.kind === "library_document" && attachment.document_id)
    .map((attachment) => ({
      ...attachment,
      data_url: undefined,
      file_path: undefined
    }));
}

function cloneAttachments(attachments: ChatAttachment[]): ChatAttachment[] {
  return attachments.map((attachment) => ({
    ...attachment,
    metadata: { ...attachment.metadata }
  }));
}

export const useKnowledgeStore = defineStore("knowledge", () => {
  const persistedState = readPersistedState();
  const sessions = ref<ChatSession[]>([]);
  const currentSessionId = ref(persistedState.currentSessionId ?? "");
  const messages = ref<ChatMessage[]>([]);
  const memorySnapshot = ref<MemorySnapshot | null>(null);
  const contextState = ref<ChatContextState | null>(null);
  const composerText = ref(persistedState.composerText ?? "");
  const draftAttachments = ref<ChatAttachment[]>(persistableAttachments(persistedState.draftAttachments ?? []));
  const selectedDocumentIds = ref<string[]>(persistedState.selectedDocumentIds ?? []);
  const uploadedTaskDocumentIds = ref<string[]>(persistedState.uploadedTaskDocumentIds ?? []);
  const loading = ref(false);
  const sending = ref(false);
  const stopping = ref(false);
  const savingReportMessageId = ref("");
  const error = ref("");
  const retrievalNotice = ref("");
  const bootstrapped = ref(false);
  const activeChatAbortController = ref<AbortController | null>(null);

  const currentSession = computed(
    () => sessions.value.find((item) => item.id === currentSessionId.value) ?? null
  );

  async function bootstrap() {
    if (bootstrapped.value && currentSessionId.value) {
      return;
    }
    loading.value = true;
    error.value = "";
    try {
      sessions.value = await listChatSessions();
      if (!sessions.value.length) {
        const session = await createChatSession({ title: "新对话" });
        sessions.value = [session];
      }
      const savedSession = sessions.value.find((item) => item.id === currentSessionId.value);
      const targetSession = savedSession ?? sessions.value[0];
      await openSession(targetSession.id, { preserveComposer: true });
      bootstrapped.value = true;
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载知识聊天失败";
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function refreshSessions() {
    sessions.value = await listChatSessions();
  }

  async function createNewSession() {
    const session = await createChatSession({ title: "新对话" });
    sessions.value = [session, ...sessions.value];
    await openSession(session.id, { preserveComposer: false });
    return session;
  }

  async function removeSession(sessionId: string) {
    loading.value = true;
    error.value = "";
    try {
      await deleteChatSession(sessionId);
      const remaining = sessions.value.filter((item) => item.id !== sessionId);
      sessions.value = remaining;
      if (currentSessionId.value !== sessionId) {
        return;
      }
      currentSessionId.value = "";
      messages.value = [];
      memorySnapshot.value = null;
      contextState.value = null;
      clearComposer();
      retrievalNotice.value = "";
      if (remaining.length) {
        await openSession(remaining[0].id, { preserveComposer: false });
      } else {
        await createNewSession();
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : "删除会话失败";
      throw err;
    } finally {
      loading.value = false;
    }
  }

  async function openSession(sessionId: string, options: { preserveComposer?: boolean } = {}) {
    loading.value = true;
    error.value = "";
    try {
      const sameSession = currentSessionId.value === sessionId;
      const detail = await getChatSessionDetail(sessionId);
      currentSessionId.value = detail.session.id;
      messages.value = detail.messages;
      memorySnapshot.value = detail.memory_snapshot;
      contextState.value = detail.context_state;
      upsertSession(detail.session);
      if (!options.preserveComposer && !sameSession) {
        clearComposer();
      }
      retrievalNotice.value = latestRetrievalNotice(detail.messages);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载会话失败";
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function queueImageAttachment(attachment: ChatAttachment) {
    draftAttachments.value = [...draftAttachments.value, attachment];
  }

  function queueLocalPdfAttachment(file: File) {
    const attachment: ChatAttachment = {
      id: crypto.randomUUID(),
      kind: "uploaded_pdf",
      display_name: file.name,
      mime_type: file.type || "application/pdf",
      status: "ready",
      metadata: {
        filename: file.name,
        size: file.size
      }
    };
    draftAttachments.value = [...draftAttachments.value, attachment];
  }

  function toggleLibraryDocument(document: LibraryDocument) {
    const exists = selectedDocumentIds.value.includes(document.id);
    if (exists) {
      selectedDocumentIds.value = selectedDocumentIds.value.filter((item) => item !== document.id);
      uploadedTaskDocumentIds.value = uploadedTaskDocumentIds.value.filter((item) => item !== document.id);
      draftAttachments.value = draftAttachments.value.filter(
        (item) => !(item.kind === "library_document" && item.document_id === document.id)
      );
      return;
    }

    selectedDocumentIds.value = [...selectedDocumentIds.value, document.id];
    draftAttachments.value = [
      ...draftAttachments.value,
      {
        id: crypto.randomUUID(),
        kind: "library_document",
        display_name: document.display_name,
        document_id: document.id,
        status: document.status,
        metadata: {
          title: document.title,
          filename: document.filename
        }
      }
    ];
  }

  function removeDraftAttachment(attachmentId: string) {
    const attachment = draftAttachments.value.find((item) => item.id === attachmentId);
    draftAttachments.value = draftAttachments.value.filter((item) => item.id !== attachmentId);
    if (attachment?.document_id) {
      selectedDocumentIds.value = selectedDocumentIds.value.filter((item) => item !== attachment.document_id);
      uploadedTaskDocumentIds.value = uploadedTaskDocumentIds.value.filter((item) => item !== attachment.document_id);
    }
  }

  function markUploadedTaskDocument(documentId: string) {
    if (!uploadedTaskDocumentIds.value.includes(documentId)) {
      uploadedTaskDocumentIds.value = [...uploadedTaskDocumentIds.value, documentId];
    }
  }

  async function sendCurrentMessage(): Promise<ChatSendResponse | null> {
    const content = composerText.value.trim();
    if (!content) {
      return null;
    }

    if (!currentSessionId.value) {
      await createNewSession();
    }

    const sessionId = currentSessionId.value;
    const attachments = cloneAttachments(draftAttachments.value);
    const selectedIds = [...selectedDocumentIds.value];
    const tempUserId = `local-user-${crypto.randomUUID()}`;
    const tempAssistantId = `local-assistant-${crypto.randomUUID()}`;
    const createdAt = new Date().toISOString();
    const optimisticUserMessage: ChatMessage = {
      id: tempUserId,
      session_id: sessionId,
      role: "user",
      content,
      status: "sending",
      citations: [],
      used_document_ids: selectedIds,
      memory_hits: [],
      attachments,
      created_at: createdAt
    };
    const optimisticAssistantMessage: ChatMessage = {
      id: tempAssistantId,
      session_id: sessionId,
      role: "assistant",
      content: "",
      status: "processing",
      retrieval_status: "skipped",
      citations: [],
      used_document_ids: [],
      memory_hits: [],
      attachments: [],
      action_status: "processing",
      created_at: createdAt
    };

    messages.value = [...messages.value, optimisticUserMessage, optimisticAssistantMessage];
    clearComposer();
    sending.value = true;
    stopping.value = false;
    error.value = "";
    retrievalNotice.value = "";
    const abortController = new AbortController();
    activeChatAbortController.value = abortController;
    const streamState: { finalResponse: ChatSendResponse | null } = { finalResponse: null };
    let streamedContent = "";
    const typewriterQueue: string[] = [];
    let typewriterTimer: number | null = null;
    let typewriterDrainResolve: (() => void) | null = null;

    const resolveTypewriterDrain = () => {
      if (typewriterQueue.length || typewriterTimer !== null || !typewriterDrainResolve) {
        return;
      }
      const resolve = typewriterDrainResolve;
      typewriterDrainResolve = null;
      resolve();
    };

    const pumpTypewriter = () => {
      const nextCharacter = typewriterQueue.shift();
      if (nextCharacter) {
        streamedContent += nextCharacter;
        updateMessage(tempAssistantId, {
          content: streamedContent,
          status: "streaming"
        });
        return;
      }
      if (typewriterTimer !== null) {
        window.clearInterval(typewriterTimer);
        typewriterTimer = null;
      }
      resolveTypewriterDrain();
    };

    const enqueueAssistantDelta = (delta: string) => {
      if (!delta) {
        return;
      }
      typewriterQueue.push(...Array.from(delta));
      if (typewriterTimer === null) {
        pumpTypewriter();
        typewriterTimer = window.setInterval(pumpTypewriter, 12);
      }
    };

    const waitForTypewriter = () => {
      if (!typewriterQueue.length && typewriterTimer === null) {
        return Promise.resolve();
      }
      return new Promise<void>((resolve) => {
        typewriterDrainResolve = resolve;
      });
    };

    const stopTypewriter = () => {
      if (typewriterTimer !== null) {
        window.clearInterval(typewriterTimer);
        typewriterTimer = null;
      }
      typewriterQueue.length = 0;
      resolveTypewriterDrain();
    };

    try {
      await streamChatMessage(sessionId, {
        content,
        attachments,
        selected_document_ids: selectedIds
      }, (event: ChatStreamEvent) => {
        if (event.type === "status") {
          updateMessage(tempAssistantId, {
            status: event.status || "processing",
            action_status: event.status || "processing"
          });
          return;
        }
        if (event.type === "assistant_delta") {
          enqueueAssistantDelta(event.delta);
          return;
        }
        if (event.type === "done") {
          streamState.finalResponse = event.response;
        }
      }, abortController.signal);

      const completedResponse = streamState.finalResponse;
      if (!completedResponse) {
        throw new Error("Chat stream ended before completion.");
      }

      if (!streamedContent && completedResponse.assistant_message.content) {
        enqueueAssistantDelta(completedResponse.assistant_message.content);
      }
      await waitForTypewriter();
      replaceOptimisticMessages(tempUserId, tempAssistantId, completedResponse);
      memorySnapshot.value = completedResponse.memory_snapshot;
      contextState.value = completedResponse.context_state;
      upsertSession(completedResponse.session);
      currentSessionId.value = completedResponse.session.id;
      retrievalNotice.value = completedResponse.assistant_message.warning || "";
      await refreshSessions();
      return completedResponse;
    } catch (err) {
      if (isAbortError(err)) {
        stopTypewriter();
        updateMessage(tempAssistantId, {
          content: streamedContent || "已停止生成。",
          status: "completed",
          action_status: "user_stopped"
        });
        retrievalNotice.value = "已停止生成。";
        return null;
      }
      error.value = err instanceof Error ? err.message : "发送消息失败";
      updateMessage(tempAssistantId, {
        content: error.value,
        status: "failed",
        action_status: "failed"
      });
      throw err;
    } finally {
      stopTypewriter();
      if (activeChatAbortController.value === abortController) {
        activeChatAbortController.value = null;
      }
      stopping.value = false;
      sending.value = false;
    }
  }

  function stopGeneration() {
    if (!sending.value || !activeChatAbortController.value) {
      return;
    }
    stopping.value = true;
    activeChatAbortController.value.abort();
  }

  async function saveAssistantMessageAsReport(message: ChatMessage) {
    if (message.role !== "assistant" || !currentSessionId.value || message.saved_report_id) {
      return null;
    }
    savingReportMessageId.value = message.id;
    error.value = "";
    try {
      const report = await saveChatMessageAsReport(currentSessionId.value, message.id);
      messages.value = messages.value.map((item) =>
        item.id === message.id
          ? {
              ...item,
              saved_report_id: report.id,
              action_status: "report_saved"
            }
          : item
      );
      return report;
    } catch (err) {
      error.value = err instanceof Error ? err.message : "保存报告失败";
      throw err;
    } finally {
      savingReportMessageId.value = "";
    }
  }

  function clearComposer() {
    composerText.value = "";
    draftAttachments.value = [];
    selectedDocumentIds.value = [];
    uploadedTaskDocumentIds.value = [];
  }

  function persistDraftState() {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        currentSessionId: currentSessionId.value,
        composerText: composerText.value,
        selectedDocumentIds: selectedDocumentIds.value,
        uploadedTaskDocumentIds: uploadedTaskDocumentIds.value,
        draftAttachments: persistableAttachments(draftAttachments.value)
      })
    );
  }

  watch(
    [currentSessionId, composerText, selectedDocumentIds, uploadedTaskDocumentIds, draftAttachments],
    persistDraftState,
    { deep: true }
  );

  function upsertSession(session: ChatSession) {
    const index = sessions.value.findIndex((item) => item.id === session.id);
    if (index === -1) {
      sessions.value = [session, ...sessions.value];
      return;
    }
    const next = [...sessions.value];
    next[index] = session;
    sessions.value = next.sort(
      (left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime()
    );
  }

  function updateMessage(messageId: string, patch: Partial<ChatMessage>) {
    messages.value = messages.value.map((message) =>
      message.id === messageId
        ? {
            ...message,
            ...patch
          }
        : message
    );
  }

  function replaceOptimisticMessages(userMessageId: string, assistantMessageId: string, response: ChatSendResponse) {
    messages.value = messages.value.map((message) => {
      if (message.id === userMessageId) {
        return response.user_message;
      }
      if (message.id === assistantMessageId) {
        return response.assistant_message;
      }
      return message;
    });
  }

  function latestRetrievalNotice(items: ChatMessage[]) {
    for (let index = items.length - 1; index >= 0; index -= 1) {
      const message = items[index];
      if (message.warning) {
        return message.warning;
      }
    }
    return "";
  }

  return {
    sessions,
    currentSessionId,
    currentSession,
    messages,
    memorySnapshot,
    contextState,
    composerText,
    draftAttachments,
    selectedDocumentIds,
    uploadedTaskDocumentIds,
    loading,
    sending,
    savingReportMessageId,
    error,
    retrievalNotice,
    stopping,
    bootstrap,
    refreshSessions,
    createNewSession,
    removeSession,
    openSession,
    queueImageAttachment,
    queueLocalPdfAttachment,
    toggleLibraryDocument,
    markUploadedTaskDocument,
    removeDraftAttachment,
    sendCurrentMessage,
    stopGeneration,
    saveAssistantMessageAsReport,
    clearComposer
  };
});

function isAbortError(err: unknown) {
  return err instanceof DOMException && err.name === "AbortError";
}
