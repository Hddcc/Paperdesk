import { defineStore } from "pinia";
import { computed, ref } from "vue";

import {
  createChatSession,
  getChatSessionDetail,
  listChatSessions,
  sendChatMessage
} from "../services/api";
import type {
  ChatAttachment,
  ChatContextState,
  ChatMessage,
  ChatSession,
  LibraryDocument,
  MemorySnapshot
} from "../types/models";

export const useKnowledgeStore = defineStore("knowledge", () => {
  const sessions = ref<ChatSession[]>([]);
  const currentSessionId = ref("");
  const messages = ref<ChatMessage[]>([]);
  const memorySnapshot = ref<MemorySnapshot | null>(null);
  const contextState = ref<ChatContextState | null>(null);
  const composerText = ref("");
  const draftAttachments = ref<ChatAttachment[]>([]);
  const selectedDocumentIds = ref<string[]>([]);
  const loading = ref(false);
  const sending = ref(false);
  const error = ref("");
  const retrievalNotice = ref("");

  const currentSession = computed(
    () => sessions.value.find((item) => item.id === currentSessionId.value) ?? null
  );

  async function bootstrap() {
    loading.value = true;
    error.value = "";
    try {
      sessions.value = await listChatSessions();
      if (!sessions.value.length) {
        const session = await createChatSession({ title: "新对话" });
        sessions.value = [session];
      }
      const targetSession = sessions.value[0];
      await openSession(targetSession.id);
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
    await openSession(session.id);
    return session;
  }

  async function openSession(sessionId: string) {
    loading.value = true;
    error.value = "";
    try {
      const detail = await getChatSessionDetail(sessionId);
      currentSessionId.value = detail.session.id;
      messages.value = detail.messages;
      memorySnapshot.value = detail.memory_snapshot;
      contextState.value = detail.context_state;
      upsertSession(detail.session);
      clearComposer();
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
    }
  }

  async function sendCurrentMessage() {
    if (!composerText.value.trim()) {
      return;
    }

    if (!currentSessionId.value) {
      await createNewSession();
    }

    sending.value = true;
    error.value = "";
    retrievalNotice.value = "";
    try {
      const response = await sendChatMessage(currentSessionId.value, {
        content: composerText.value.trim(),
        attachments: draftAttachments.value,
        selected_document_ids: selectedDocumentIds.value
      });
      messages.value = [...messages.value, response.user_message, response.assistant_message];
      memorySnapshot.value = response.memory_snapshot;
      contextState.value = response.context_state;
      upsertSession(response.session);
      currentSessionId.value = response.session.id;
      retrievalNotice.value = response.assistant_message.warning || "";
      clearComposer();
      await refreshSessions();
    } catch (err) {
      error.value = err instanceof Error ? err.message : "发送消息失败";
      throw err;
    } finally {
      sending.value = false;
    }
  }

  function clearComposer() {
    composerText.value = "";
    draftAttachments.value = [];
    selectedDocumentIds.value = [];
  }

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
    loading,
    sending,
    error,
    retrievalNotice,
    bootstrap,
    refreshSessions,
    createNewSession,
    openSession,
    queueImageAttachment,
    queueLocalPdfAttachment,
    toggleLibraryDocument,
    removeDraftAttachment,
    sendCurrentMessage,
    clearComposer
  };
});
