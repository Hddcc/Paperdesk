import { defineStore } from "pinia";
import { computed, ref, watch } from "vue";

import {
  createChatSession,
  deleteChatSession,
  getChatSessionDetail,
  getWorkbenchCapabilities,
  getWorkbenchConfig,
  getWorkbenchSessionFiles,
  listChatSessions,
  saveChatMessageAsReport,
  streamChatMessage,
  uploadWorkbenchSessionFile
} from "../services/api";
import type {
  ChatAttachment,
  ChatContextState,
  ChatMessage,
  ChatSendResponse,
  ChatStreamEvent,
  ChatSession,
  LibraryDocument,
  MemorySnapshot,
  SlashCommandOption,
  WorkbenchCapabilitiesResponse,
  WorkbenchConfigResponse,
  WorkbenchFileContextResponse,
  WorkbenchFileAsset,
  WorkspaceFileFormat
} from "../types/models";

type SelectableLibraryDocument = Pick<LibraryDocument, "id" | "display_name" | "title" | "filename" | "status">;

const STORAGE_KEY = "paperdesk-knowledge-draft-state";

interface BrowserSaveFilePickerOptions {
  suggestedName?: string;
  types?: Array<{
    description?: string;
    accept: Record<string, string[]>;
  }>;
}

interface BrowserFileSystemWritableFileStream {
  write(data: Blob): Promise<void>;
  close(): Promise<void>;
}

interface BrowserFileSystemFileHandle {
  createWritable(): Promise<BrowserFileSystemWritableFileStream>;
}

declare global {
  interface Window {
    showSaveFilePicker?: (options?: BrowserSaveFilePickerOptions) => Promise<BrowserFileSystemFileHandle>;
  }
}

export const SLASH_COMMANDS: SlashCommandOption[] = [
  {
    id: "summary",
    label: "/summary",
    group: "论文",
    description: "总结选中的论文。",
    intent_hint: "paper_summary",
    min_documents: 1,
    default_prompt: "请总结所选论文。"
  },
  {
    id: "compare",
    label: "/compare",
    group: "论文",
    description: "对比选中的论文。",
    intent_hint: "paper_compare",
    min_documents: 2,
    default_prompt: "请对比所选论文。"
  },
  {
    id: "tag",
    label: "/tag",
    group: "论文库",
    description: "查询或整理标签和分类。",
    intent_hint: "tag_query_or_write",
    default_prompt: "请处理标签或分类相关问题。",
    warning: "标签修改会进入确认流程。"
  },
  {
    id: "library",
    label: "/library",
    group: "论文库",
    description: "查询论文库数量、状态和标签。",
    intent_hint: "library_query",
    default_prompt: "请查询当前论文库状态。"
  },
  {
    id: "help",
    label: "/help",
    group: "帮助",
    description: "显示可用的斜杠命令。",
    intent_hint: "help",
    default_prompt: "显示可用命令说明。",
    local_only: true
  }
];

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
  const workbenchConfig = ref<WorkbenchConfigResponse | null>(null);
  const workbenchCapabilities = ref<WorkbenchCapabilitiesResponse | null>(null);
  const workbenchFileContext = ref<WorkbenchFileContextResponse | null>(null);
  const composerText = ref(persistedState.composerText ?? "");
  const activeSlashCommand = ref<SlashCommandOption | null>(null);
  const slashCommandMenuOpen = ref(false);
  const draftAttachments = ref<ChatAttachment[]>(persistableAttachments(persistedState.draftAttachments ?? []));
  const selectedDocumentIds = ref<string[]>(persistedState.selectedDocumentIds ?? []);
  const selectedFileIds = ref<string[]>([]);
  const uploadedTaskDocumentIds = ref<string[]>(persistedState.uploadedTaskDocumentIds ?? []);
  const selectedAgentProfileId = ref("paper_qa");
  const selectedModelId = ref("");
  const loading = ref(false);
  const isWorkbenchLoading = ref(false);
  const isCapabilitiesLoading = ref(false);
  const isUploadingSessionFile = ref(false);
  const sending = ref(false);
  const stopping = ref(false);
  const savingReportMessageId = ref("");
  const isSavingWorkspaceFile = ref<Record<string, boolean>>({});
  const workspaceFileSaveError = ref<string | null>(null);
  const workspaceFileSaveErrorMessageId = ref("");
  const error = ref("");
  const sessionFileUploadError = ref("");
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
      await loadWorkbenchConfig();
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
      await refreshWorkbenchFileContext();
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

  function toggleLibraryDocument(document: SelectableLibraryDocument) {
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

  function toggleSessionFile(fileId: string) {
    const file = findWorkbenchFile(fileId);
    const normalizedFileId = file ? getWorkbenchFileId(file) : fileId;
    if (!file || !isSelectableSessionFile(file)) {
      selectedFileIds.value = selectedFileIds.value.filter((item) => item !== normalizedFileId);
      return;
    }
    if (selectedFileIds.value.includes(normalizedFileId)) {
      selectedFileIds.value = selectedFileIds.value.filter((item) => item !== normalizedFileId);
      return;
    }
    selectedFileIds.value = [...selectedFileIds.value, normalizedFileId];
  }

  function addSelectedSessionFile(file: WorkbenchFileAsset) {
    if (!isSelectableSessionFile(file)) {
      return;
    }
    const fileId = getWorkbenchFileId(file);
    if (!selectedFileIds.value.includes(fileId)) {
      selectedFileIds.value = [...selectedFileIds.value, fileId];
    }
  }

  function clearSelectedFiles() {
    selectedFileIds.value = [];
  }

  function isSessionFileSelected(fileId: string) {
    return selectedFileIds.value.includes(fileId);
  }

  function markUploadedTaskDocument(documentId: string) {
    if (!uploadedTaskDocumentIds.value.includes(documentId)) {
      uploadedTaskDocumentIds.value = [...uploadedTaskDocumentIds.value, documentId];
    }
  }

  async function loadWorkbenchConfig() {
    isWorkbenchLoading.value = true;
    try {
      const config = await getWorkbenchConfig();
      workbenchConfig.value = config;
      selectedAgentProfileId.value = config.agent_profiles[0]?.id ?? "paper_qa";
      selectedModelId.value = config.current_model;
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载助手配置失败";
    } finally {
      isWorkbenchLoading.value = false;
    }
  }

  async function loadWorkbenchCapabilities() {
    isCapabilitiesLoading.value = true;
    try {
      workbenchCapabilities.value = await getWorkbenchCapabilities();
    } catch (err) {
      workbenchCapabilities.value = null;
      error.value = err instanceof Error ? err.message : "加载助手能力失败";
    } finally {
      isCapabilitiesLoading.value = false;
    }
  }

  async function refreshWorkbenchFileContext() {
    if (!currentSessionId.value) {
      workbenchFileContext.value = null;
      return;
    }
    isWorkbenchLoading.value = true;
    try {
      workbenchFileContext.value = await getWorkbenchSessionFiles(currentSessionId.value);
    } catch (err) {
      workbenchFileContext.value = null;
      error.value = err instanceof Error ? err.message : "加载论文库与文件状态失败";
    } finally {
      isWorkbenchLoading.value = false;
    }
  }

  async function uploadSessionFile(file: File) {
    if (!currentSessionId.value) {
      await createNewSession();
    }
    isUploadingSessionFile.value = true;
    sessionFileUploadError.value = "";
    try {
      const uploadedFile = await uploadWorkbenchSessionFile(currentSessionId.value, file);
      await refreshWorkbenchFileContext();
      return uploadedFile;
    } catch (err) {
      sessionFileUploadError.value = err instanceof Error ? err.message : "上传会话文件失败";
      throw err;
    } finally {
      isUploadingSessionFile.value = false;
    }
  }

  function setAgentProfile(id: string) {
    selectedAgentProfileId.value = id;
  }

  function setModel(id: string) {
    selectedModelId.value = id;
  }

  function setSlashCommand(command: SlashCommandOption | null) {
    activeSlashCommand.value = command;
  }

  function clearSlashCommand() {
    activeSlashCommand.value = null;
    slashCommandMenuOpen.value = false;
  }

  function applySlashCommand(command: SlashCommandOption) {
    activeSlashCommand.value = command;
    slashCommandMenuOpen.value = false;
    composerText.value = composerText.value.replace(/(^|\s)\/[^\s]*$/, "$1").trimEnd();
    if (command.local_only) {
      retrievalNotice.value = slashHelpText();
    }
  }

  async function sendCurrentMessage(): Promise<ChatSendResponse | null> {
    const command = activeSlashCommand.value;
    const content = composerText.value.trim() || command?.default_prompt || "";
    if (!content) {
      return null;
    }

    if (command?.local_only) {
      retrievalNotice.value = slashHelpText();
      composerText.value = "";
      clearSlashCommand();
      return null;
    }

    if (!currentSessionId.value) {
      await createNewSession();
    }

    const sessionId = currentSessionId.value;
    const attachments = cloneAttachments(draftAttachments.value);
    const selectedIds = [...selectedDocumentIds.value];
    const selectedFileIdsForMessage = [...selectedFileIds.value];
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
      selected_file_ids: selectedFileIdsForMessage,
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
      used_file_ids: [],
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
        selected_document_ids: selectedIds,
        selected_file_ids: selectedFileIdsForMessage,
        agent_profile_id: selectedAgentProfileId.value || null,
        model_id: selectedModelId.value || null,
        command: command?.id ?? null,
        intent_hint: command?.intent_hint ?? null
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
      await refreshWorkbenchFileContext();
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

  async function saveAssistantMessageAsWorkspaceFile(
    messageId: string,
    suggestedFilename: string,
    format: WorkspaceFileFormat
  ) {
    const message = messages.value.find((item) => item.id === messageId);
    if (!message || message.role !== "assistant") {
      workspaceFileSaveError.value = "只能保存助手消息。";
      workspaceFileSaveErrorMessageId.value = messageId;
      return null;
    }
    if (!currentSessionId.value) {
      workspaceFileSaveError.value = "当前会话不可用，请稍后重试。";
      workspaceFileSaveErrorMessageId.value = messageId;
      return null;
    }
    isSavingWorkspaceFile.value = {
      ...isSavingWorkspaceFile.value,
      [messageId]: true
    };
    workspaceFileSaveError.value = null;
    workspaceFileSaveErrorMessageId.value = messageId;
    try {
      await saveTextWithBrowserPicker(message.content, suggestedFilename, format);
      workspaceFileSaveErrorMessageId.value = "";
      return null;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        workspaceFileSaveErrorMessageId.value = "";
        return null;
      }
      workspaceFileSaveError.value = formatWorkspaceFileSaveError(err);
      throw err;
    } finally {
      isSavingWorkspaceFile.value = {
        ...isSavingWorkspaceFile.value,
        [messageId]: false
      };
    }
  }

  function clearComposer() {
    composerText.value = "";
    activeSlashCommand.value = null;
    slashCommandMenuOpen.value = false;
    draftAttachments.value = [];
    selectedDocumentIds.value = [];
    selectedFileIds.value = [];
    uploadedTaskDocumentIds.value = [];
  }

  function getWorkbenchFileId(file: WorkbenchFileAsset) {
    return file.file_id || file.id;
  }

  function findWorkbenchFile(fileId: string) {
    const files = [
      ...(workbenchFileContext.value?.session_files ?? []),
      ...(workbenchFileContext.value?.workspace_files ?? [])
    ];
    return files.find((file) => getWorkbenchFileId(file) === fileId || file.id === fileId || file.file_id === fileId);
  }

  function isSelectableSessionFile(file: WorkbenchFileAsset) {
    return file.status === "ready" && file.text_extract_status === "ready";
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
    [
      currentSessionId,
      composerText,
      selectedDocumentIds,
      uploadedTaskDocumentIds,
      draftAttachments
    ],
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

  function slashHelpText() {
    return SLASH_COMMANDS
      .filter((command) => command.id !== "help")
      .map((command) => `${command.label} - ${command.description}`)
      .join("\n");
  }

  return {
    sessions,
    currentSessionId,
    currentSession,
    messages,
    memorySnapshot,
    contextState,
    workbenchConfig,
    workbenchCapabilities,
    workbenchFileContext,
    composerText,
    activeSlashCommand,
    slashCommandMenuOpen,
    draftAttachments,
    selectedDocumentIds,
    selectedFileIds,
    uploadedTaskDocumentIds,
    selectedAgentProfileId,
    selectedModelId,
    loading,
    isWorkbenchLoading,
    isCapabilitiesLoading,
    isUploadingSessionFile,
    sending,
    savingReportMessageId,
    isSavingWorkspaceFile,
    workspaceFileSaveError,
    workspaceFileSaveErrorMessageId,
    error,
    sessionFileUploadError,
    retrievalNotice,
    stopping,
    bootstrap,
    refreshSessions,
    loadWorkbenchConfig,
    loadWorkbenchCapabilities,
    refreshWorkbenchFileContext,
    uploadSessionFile,
    createNewSession,
    removeSession,
    openSession,
    setAgentProfile,
    setModel,
    setSlashCommand,
    clearSlashCommand,
    applySlashCommand,
    queueImageAttachment,
    toggleLibraryDocument,
    toggleSessionFile,
    addSelectedSessionFile,
    clearSelectedFiles,
    isSessionFileSelected,
    markUploadedTaskDocument,
    removeDraftAttachment,
    sendCurrentMessage,
    stopGeneration,
    saveAssistantMessageAsReport,
    saveAssistantMessageAsWorkspaceFile,
    clearComposer
  };
});

async function saveTextWithBrowserPicker(
  content: string,
  suggestedFilename: string,
  format: WorkspaceFileFormat
) {
  const normalizedFilename = normalizeSuggestedFilename(suggestedFilename, format);
  const pickerMimeType = format === "txt" ? "text/plain" : "text/markdown";
  const blobMimeType = `${pickerMimeType};charset=utf-8`;
  if (window.showSaveFilePicker) {
    const handle = await window.showSaveFilePicker({
      suggestedName: normalizedFilename,
      types: [
        {
          description: format === "txt" ? "TXT 文本" : "Markdown 文档",
          accept: {
            [pickerMimeType]: [`.${format}`]
          }
        }
      ]
    });
    const writable = await handle.createWritable();
    await writable.write(new Blob([content], { type: blobMimeType }));
    await writable.close();
    return;
  }

  const blob = new Blob([content], { type: blobMimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = normalizedFilename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function normalizeSuggestedFilename(filename: string, format: WorkspaceFileFormat) {
  const fallback = `answer.${format}`;
  const cleaned = filename.trim() || fallback;
  if (/\.(md|txt)$/i.test(cleaned)) {
    return cleaned.replace(/\.(md|txt)$/i, `.${format}`);
  }
  return `${cleaned}.${format}`;
}

function isAbortError(err: unknown) {
  return err instanceof DOMException && err.name === "AbortError";
}

function formatWorkspaceFileSaveError(err: unknown) {
  const message = err instanceof Error ? err.message : "";
  if (message.includes("Only assistant messages")) {
    return "只能保存助手消息。";
  }
  if (message.includes("content is empty")) {
    return "这条消息没有可保存的内容。";
  }
  if (message.includes("already exists")) {
    return "该文件已存在，请换一个文件名。";
  }
  if (
    message.includes("path separators") ||
    message.includes("invalid") ||
    message.includes("not allowed") ||
    message.includes("unsafe") ||
    message.includes("Security")
  ) {
    return "文件名不安全，请只输入普通文件名。";
  }
  if (
    message.includes("Unsupported workspace file format") ||
    message.includes("Unsupported workspace file extension") ||
    message.includes("does not match")
  ) {
    return "当前只支持保存为 Markdown 或 TXT，请检查扩展名和格式。";
  }
  if (message.includes("not found")) {
    return "当前会话或消息不存在，请刷新后重试。";
  }
  return message || "保存文件失败，请稍后重试。";
}
