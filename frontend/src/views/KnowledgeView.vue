<template>
  <section class="page-shell knowledge-chat-page">
    <aside class="chat-session-panel panel">
      <header class="chat-panel-head">
        <div>
          <p class="eyebrow">知识对话</p>
          <h2>知识库对话</h2>
        </div>
        <button class="button-secondary" :disabled="store.loading" @click="store.createNewSession">
          <Plus :size="16" />
          新建对话
        </button>
      </header>

      <div class="panel-body panel-scroll">
        <ul class="chat-session-list">
          <li v-for="session in store.sessions" :key="session.id" class="chat-session-row">
            <button
              class="chat-session-button"
              :class="{ 'chat-session-active': session.id === store.currentSessionId }"
              @click="store.openSession(session.id)"
              :title="sessionSummary(session)"
            >
              <span class="chat-session-title">{{ sessionSummary(session) }}</span>
            </button>
            <button
              class="chat-session-delete"
              type="button"
              aria-label="删除对话"
              title="删除对话"
              :disabled="store.loading"
              @click.stop="store.removeSession(session.id)"
            >
              <Trash2 :size="15" aria-hidden="true" />
            </button>
          </li>
        </ul>

      </div>
    </aside>

    <article class="panel chat-stage">
      <header class="chat-stage-head">
        <div>
          <p class="eyebrow">PaperDesk 助手</p>
          <h2>{{ store.currentSession?.title || "新对话" }}</h2>
          <p>可直接提问，也可以上传文件或选择论文材料，再生成问答、总结、综述与对比结果。</p>
        </div>
      </header>

      <div ref="messagePanelRef" class="panel-body panel-scroll chat-message-panel">
        <div v-if="store.loading" class="empty-state">正在加载会话…</div>

        <div v-else-if="!store.messages.length" class="chat-empty-state">
        <div class="chat-empty-copy">
            <h3>开始一轮新的知识对话</h3>
            <p>你可以直接提问，也可以上传文件或从论文库里选择已入库论文。</p>
          </div>
        </div>

        <ul v-else class="chat-message-list">
          <li
            v-for="message in store.messages"
            :key="message.id"
            class="chat-message-item"
            :data-role="message.role"
          >
            <div class="chat-message-bubble">
              <div class="chat-message-meta">
                <strong>{{ formatRole(message.role) }}</strong>
                <span>{{ formatTime(message.created_at) }}</span>
              </div>

              <div v-if="message.attachments.length" class="chat-attachment-grid">
                <article
                  v-for="attachment in message.attachments"
                  :key="attachment.id"
                  class="chat-attachment-card"
                >
                  <img
                    v-if="attachment.kind === 'image' && attachment.data_url"
                    :src="attachment.data_url"
                    :alt="attachment.display_name"
                    class="chat-attachment-image"
                  />
                  <div>
                    <strong>{{ attachment.display_name }}</strong>
                    <p class="card-meta">
                      {{ formatAttachmentKind(attachment.kind) }}
                      <span v-if="attachment.status"> · {{ attachment.status }}</span>
                    </p>
                  </div>
                </article>
              </div>

              <MarkdownPreview
                v-if="message.role === 'assistant' && message.content"
                :markdown="message.content"
              />
              <div v-if="message.role === 'assistant' && isMessageProcessing(message)" class="chat-processing-indicator">
                <span>正在处理</span>
                <i></i>
                <i></i>
                <i></i>
              </div>
              <p v-if="message.role !== 'assistant'" class="chat-message-text">{{ message.content }}</p>

              <p v-if="message.warning" class="hint-text">{{ message.warning }}</p>

              <div class="chat-message-actions">
                <button
                  class="button-secondary message-action-button"
                  :disabled="!message.content.trim()"
                  @click="copyMessage(message)"
                >
                  <Copy :size="15" />
                  {{ copiedMessageId === message.id ? "已复制" : "复制" }}
                </button>
                <button
                  v-if="message.role === 'assistant'"
                  class="button-secondary message-action-button"
                  :disabled="isMessageProcessing(message) || !message.content.trim() || Boolean(message.saved_report_id) || store.savingReportMessageId === message.id"
                  @click="saveMessageReport(message)"
                >
                  <Save :size="15" />
                  {{
                    message.saved_report_id
                      ? "已保存"
                      : store.savingReportMessageId === message.id
                        ? "保存中..."
                        : "保存为报告"
                  }}
                </button>
              </div>

            </div>
          </li>
        </ul>
      </div>

      <div class="chat-composer">
        <div v-if="store.retrievalNotice" class="hint-text">{{ store.retrievalNotice }}</div>
        <p v-if="store.error" class="error-text">{{ store.error }}</p>
        <p v-if="documentStore.error" class="error-text">{{ documentStore.error }}</p>
        <p v-if="store.sessionFileUploadError" class="error-text">{{ store.sessionFileUploadError }}</p>

        <div v-if="visibleDraftAttachments.length" class="draft-strip">
          <div class="memory-chip-list">
            <button
              v-for="attachment in visibleDraftAttachments"
              :key="attachment.id"
              class="draft-chip"
              @click="store.removeDraftAttachment(attachment.id)"
            >
              <span>{{ attachment.display_name }}</span>
              <small>{{ formatAttachmentKind(attachment.kind) }}</small>
            </button>
          </div>
        </div>

        <div v-if="showSlashMenu" class="slash-command-menu" role="listbox" aria-label="Slash commands">
          <button
            v-for="(command, index) in filteredSlashCommands"
            :key="command.id"
            class="slash-command-row"
            :class="{ 'slash-command-row-active': index === slashCommandIndex }"
            type="button"
            role="option"
            :aria-selected="index === slashCommandIndex"
            @mousedown.prevent="selectSlashCommand(command)"
          >
            <span class="slash-command-label">{{ command.label }}</span>
            <span class="slash-command-copy">
              <strong>{{ command.description }}</strong>
              <small>{{ command.group }}</small>
            </span>
          </button>
        </div>

        <div class="composer-shell">
          <div
            class="composer-resize-handle"
            role="separator"
            aria-label="调整输入框高度"
            aria-orientation="horizontal"
            @pointerdown="startComposerResize"
          ></div>
          <div v-if="selectedLibraryDraftAttachments.length" class="selected-document-strip" aria-label="已选论文">
            <article
              v-for="attachment in selectedLibraryDraftAttachments"
              :key="attachment.id"
              class="selected-document-chip"
              :title="attachment.display_name"
            >
              <span class="selected-document-icon" aria-hidden="true">
                <FileText :size="18" />
              </span>
              <span class="selected-document-copy">
                <strong>{{ attachment.display_name }}</strong>
                <small>
                  {{ formatAttachmentKind(attachment.kind) }}
                  <span v-if="attachment.status"> · {{ formatDocumentStatus(attachment.status) }}</span>
                </small>
              </span>
              <button
                class="selected-document-remove"
                type="button"
                :aria-label="`移除 ${attachment.display_name}`"
                @click="store.removeDraftAttachment(attachment.id)"
              >
                <X :size="14" />
              </button>
            </article>
          </div>
          <p v-if="mixedSelectionHint" class="mixed-selection-hint">{{ mixedSelectionHint }}</p>
          <div v-if="selectedSessionFileChips.length" class="selected-document-strip selected-file-strip" aria-label="已选普通文件">
            <article
              v-for="file in selectedSessionFileChips"
              :key="getSessionFileId(file)"
              class="selected-document-chip selected-file-chip"
              :title="file.display_name || file.filename"
            >
              <span class="selected-document-icon selected-file-icon" aria-hidden="true">
                {{ formatFileKind(file.kind) }}
              </span>
              <span class="selected-document-copy">
                <strong>{{ file.display_name || file.filename }}</strong>
                <small>{{ formatFileKind(file.kind) }} · {{ formatBytes(file.size_bytes) }}</small>
              </span>
              <button
                class="selected-document-remove"
                type="button"
                :aria-label="`移除 ${file.display_name || file.filename}`"
                @click="store.toggleSessionFile(getSessionFileId(file))"
              >
                <X :size="14" />
              </button>
            </article>
          </div>
          <button class="composer-plus" aria-label="上传" title="上传" @click="toggleAttachMenu">
            <Paperclip :size="19" />
          </button>
          <div v-if="showAttachMenu" class="attach-popover" role="menu" aria-label="添加材料">
            <button type="button" role="menuitem" @click="triggerFileInput">
              <Paperclip :size="16" />
              <span>添加文件</span>
            </button>
            <button type="button" role="menuitem" @click="toggleDocumentPicker">
              <FileText :size="16" />
              <span>论文库</span>
            </button>
          </div>
          <section v-if="showDocumentPicker" class="document-picker-sheet" aria-label="论文库">
            <header class="sheet-head">
              <div>
                <p class="eyebrow">论文库</p>
                <h3>选择论文</h3>
              </div>
              <button class="icon-button" type="button" aria-label="关闭论文库" title="关闭" @click="showDocumentPicker = false">
                <X :size="15" />
              </button>
            </header>
            <div v-if="store.isWorkbenchLoading && !workbenchDocuments.length" class="empty-state">
              正在加载论文库...
            </div>
            <div v-else-if="workbenchDocuments.length" class="document-pick-grid">
              <button
                v-for="document in workbenchDocuments"
                :key="document.id"
                class="pick-card pick-card-button"
                type="button"
                :class="{ 'task-card-active': isWorkbenchDocumentSelected(document.id) }"
                :disabled="document.status !== 'ready'"
                :title="document.display_name"
                @click="toggleWorkbenchDocument(document)"
              >
                <strong class="card-title">{{ document.display_name }}</strong>
                <span class="card-meta">{{ document.title || document.filename }}</span>
                <span class="status-badge" :data-status="document.status">
                  {{ formatDocumentStatus(document.status) }}
                </span>
              </button>
            </div>
            <p v-else class="empty-state">
              暂无库内论文。
            </p>
          </section>
          <div v-if="store.activeSlashCommand" class="slash-command-token">
            <span>{{ store.activeSlashCommand.label }}</span>
            <button type="button" aria-label="Clear slash command" @click="store.clearSlashCommand">
              <X :size="13" />
            </button>
          </div>
          <p v-if="slashCommandHint" class="slash-command-hint">{{ slashCommandHint }}</p>
          <div v-if="store.activeSlashCommand?.id === 'help'" class="slash-help-card">
            <strong>Slash commands</strong>
            <span v-for="command in slashCommands" :key="command.id">
              {{ command.label }} - {{ command.description }}
            </span>
          </div>
          <textarea
            v-model="store.composerText"
            class="chat-textarea"
            :style="{ height: `${composerHeight}px` }"
            placeholder="输入问题，或上传 PDF / TXT / MD / DOCX 后发送。"
            @keydown="handleKeydown"
            @input="handleComposerInput"
          />
          <button
            class="button-primary composer-send"
            :disabled="!store.sending && !canSendMessage"
            @click="handleComposerAction"
          >
            <StopCircle v-if="store.sending" :size="16" />
            <SendHorizontal v-else :size="16" />
            {{ store.sending ? (store.stopping ? "停止中..." : "停止") : "发送" }}
          </button>
        </div>
      </div>
    </article>

    <input
      ref="fileInputRef"
      hidden
      type="file"
      accept=".pdf,.txt,.md,.docx"
      multiple
      @change="handleFilesSelected"
    />
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Copy, FileText, Paperclip, Plus, Save, SendHorizontal, StopCircle, Trash2, X } from "lucide-vue-next";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { useDocumentStore } from "../stores/documents";
import { SLASH_COMMANDS, useKnowledgeStore } from "../stores/knowledge";
import type {
  ChatMessage,
  ChatMessageRole,
  ChatSession,
  SlashCommandOption,
  WorkbenchFileAsset,
  WorkbenchFileItem
} from "../types/models";

const store = useKnowledgeStore();
const documentStore = useDocumentStore();

const messagePanelRef = ref<HTMLElement | null>(null);
const fileInputRef = ref<HTMLInputElement | null>(null);
const copiedMessageId = ref("");
const showAttachMenu = ref(false);
const showDocumentPicker = ref(false);
const composerHeight = ref(112);
const resizeStartY = ref(0);
const resizeStartHeight = ref(0);
const resizingComposer = ref(false);
const sessionFileMaxBytes = 5 * 1024 * 1024;
const allowedSessionFileExtensions = new Set(["txt", "md", "docx"]);
const allowedUploadExtensions = new Set(["pdf", ...allowedSessionFileExtensions]);

const slashCommands = SLASH_COMMANDS;
const slashCommandIndex = ref(0);
const slashQuery = computed(() => {
  const match = store.composerText.match(/(?:^|\s)(\/[^\s]*)$/);
  return match ? match[1].slice(1).toLowerCase() : "";
});
const filteredSlashCommands = computed(() => {
  const query = slashQuery.value;
  if (!query) {
    return slashCommands;
  }
  return slashCommands.filter((command) => command.id.startsWith(query));
});
const showSlashMenu = computed(() =>
  store.slashCommandMenuOpen && !store.activeSlashCommand && filteredSlashCommands.value.length > 0
);
const canSendMessage = computed(() => Boolean(store.composerText.trim() || store.activeSlashCommand));
const slashCommandHint = computed(() => {
  const command = store.activeSlashCommand;
  if (!command) {
    return "";
  }
  if (command.id === "compare" && store.selectedDocumentIds.length < 2) {
    return "建议先选择至少 2 篇 ready 论文；也可以继续发送，让后端按自然语言判断。";
  }
  if (command.warning) {
    return command.warning;
  }
  return "";
});
const workbenchDocuments = computed<WorkbenchFileItem[]>(() =>
  store.workbenchFileContext?.library_documents.length
    ? store.workbenchFileContext.library_documents
    : documentStore.documents
);
const workbenchSessionFiles = computed<WorkbenchFileAsset[]>(() => {
  const files = store.workbenchFileContext?.session_files ?? [];
  const workspaceFiles = store.workbenchFileContext?.workspace_files ?? [];
  const seen = new Set<string>();
  return [...files, ...workspaceFiles].filter((file) => {
    const id = file.file_id || file.id;
    if (seen.has(id)) {
      return false;
    }
    seen.add(id);
    return true;
  });
});
const selectedSessionFileChips = computed(() =>
  store.selectedFileIds
    .map((fileId) => workbenchSessionFiles.value.find((file) => getSessionFileId(file) === fileId))
    .filter((file): file is WorkbenchFileAsset => Boolean(file))
);
const mixedSelectionHint = computed(() =>
  store.selectedDocumentIds.length && store.selectedFileIds.length
    ? "当前后端建议普通文件和论文库文件分开提问。"
    : ""
);
const visibleDraftAttachments = computed(() =>
  store.draftAttachments.filter((attachment) => attachment.kind !== "library_document")
);
const selectedLibraryDraftAttachments = computed(() =>
  store.draftAttachments
    .filter((attachment) => attachment.kind === "library_document")
    .map((attachment) => {
      const document = documentStore.documents.find((item) => item.id === attachment.document_id);
      if (!document) {
        return attachment;
      }
      return {
        ...attachment,
        display_name: document.display_name || attachment.display_name,
        status: document.status,
        metadata: {
          ...attachment.metadata,
          title: document.title,
          filename: document.filename
        }
      };
    })
);
const latestMessageSignature = computed(() => {
  const latestMessage = store.messages[store.messages.length - 1];
  return `${store.messages.length}:${latestMessage?.id || ""}:${latestMessage?.content.length || 0}:${store.sending}`;
});

onMounted(async () => {
  await Promise.all([documentStore.refreshDocuments(), store.bootstrap()]);
});

watch(latestMessageSignature, () => {
  void scrollMessagesToBottom();
}, { flush: "post" });

async function scrollMessagesToBottom() {
  await nextTick();
  if (messagePanelRef.value) {
    messagePanelRef.value.scrollTop = messagePanelRef.value.scrollHeight;
  }
}

function toggleAttachMenu() {
  showAttachMenu.value = !showAttachMenu.value;
  if (showAttachMenu.value) {
    showDocumentPicker.value = false;
  }
}

function triggerFileInput() {
  showAttachMenu.value = false;
  fileInputRef.value?.click();
}

function toggleDocumentPicker() {
  showAttachMenu.value = false;
  showDocumentPicker.value = !showDocumentPicker.value;
  if (showDocumentPicker.value) {
    void store.refreshWorkbenchFileContext();
  }
}

async function uploadPdfToLibrary(file: File) {
  try {
    const document = await documentStore.addDocument(file);
    if (document) {
      store.toggleLibraryDocument(document);
      store.markUploadedTaskDocument(document.id);
      await store.refreshWorkbenchFileContext();
    }
  } catch {
    store.queueLocalPdfAttachment(file);
  }
}

async function uploadComposerFile(file: File) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (!allowedUploadExtensions.has(extension)) {
    store.sessionFileUploadError = "支持上传 PDF、TXT、MD、DOCX 文件。";
    return;
  }
  if (extension === "pdf") {
    await uploadPdfToLibrary(file);
    return;
  }
  if (file.size > sessionFileMaxBytes) {
    store.sessionFileUploadError = "文件超过 5MB，请选择更小的 txt、md 或 docx 文件。";
    return;
  }
  try {
    const uploadedFile = await store.uploadSessionFile(file);
    if (isSessionFileSelectable(uploadedFile)) {
      store.addSelectedSessionFile(uploadedFile);
      return;
    }
    store.sessionFileUploadError = `${uploadedFile.display_name || uploadedFile.filename} 已上传，${sessionFileUnavailableReason(uploadedFile)}`;
  } catch {
    // Store owns the visible upload error.
  }
}

async function handleFilesSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const files = Array.from(input.files ?? []);
  input.value = "";
  if (!files.length) {
    return;
  }
  store.sessionFileUploadError = "";
  for (const file of files) {
    await uploadComposerFile(file);
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (store.slashCommandMenuOpen) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveSlashCommandSelection(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      moveSlashCommandSelection(-1);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      store.slashCommandMenuOpen = false;
      return;
    }
    if (event.key === "Enter" && !event.shiftKey && showSlashMenu.value) {
      event.preventDefault();
      const command = filteredSlashCommands.value[slashCommandIndex.value];
      if (command) {
        selectSlashCommand(command);
      }
      return;
    }
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!store.sending && canSendMessage.value) {
      void sendChatMessage();
    }
  }
}

function handleComposerInput() {
  const hasSlashQuery = /(?:^|\s)\/[^\s]*$/.test(store.composerText);
  store.slashCommandMenuOpen = hasSlashQuery && !store.activeSlashCommand;
  if (store.slashCommandMenuOpen) {
    slashCommandIndex.value = 0;
  }
}

function moveSlashCommandSelection(direction: number) {
  const count = filteredSlashCommands.value.length;
  if (!count) {
    slashCommandIndex.value = 0;
    return;
  }
  slashCommandIndex.value = (slashCommandIndex.value + direction + count) % count;
}

function selectSlashCommand(command: SlashCommandOption) {
  store.applySlashCommand(command);
  slashCommandIndex.value = 0;
}

function startComposerResize(event: PointerEvent) {
  resizingComposer.value = true;
  resizeStartY.value = event.clientY;
  resizeStartHeight.value = composerHeight.value;
  window.addEventListener("pointermove", resizeComposer);
  window.addEventListener("pointerup", stopComposerResize);
  window.addEventListener("pointercancel", stopComposerResize);
}

function resizeComposer(event: PointerEvent) {
  if (!resizingComposer.value) {
    return;
  }
  const maxHeight = Math.min(Math.floor(window.innerHeight * 0.5), 420);
  const nextHeight = resizeStartHeight.value + resizeStartY.value - event.clientY;
  composerHeight.value = Math.max(92, Math.min(maxHeight, nextHeight));
}

function stopComposerResize() {
  resizingComposer.value = false;
  window.removeEventListener("pointermove", resizeComposer);
  window.removeEventListener("pointerup", stopComposerResize);
  window.removeEventListener("pointercancel", stopComposerResize);
}

onBeforeUnmount(() => {
  store.stopGeneration();
  stopComposerResize();
});

function handleComposerAction() {
  if (store.sending) {
    store.stopGeneration();
    return;
  }
  void sendChatMessage();
}

async function sendChatMessage() {
  const response = await store.sendCurrentMessage();
  if (response?.library_mutated) {
    await Promise.all([documentStore.refreshDocuments(), documentStore.refreshCategories()]);
    await store.refreshWorkbenchFileContext();
  }
}

function toggleWorkbenchDocument(document: WorkbenchFileItem) {
  if (document.status !== "ready") {
    return;
  }
  store.toggleLibraryDocument(document);
}

function isWorkbenchDocumentSelected(documentId: string) {
  return store.selectedDocumentIds.includes(documentId);
}

function getSessionFileId(file: WorkbenchFileAsset) {
  return file.file_id || file.id;
}

function isSessionFileSelectable(file: WorkbenchFileAsset) {
  return file.status === "ready" && file.text_extract_status === "ready";
}

function isMessageProcessing(message: ChatMessage) {
  return message.role === "assistant" && (message.status === "processing" || (message.status === "streaming" && !message.content.trim()));
}

async function copyMessage(message: ChatMessage) {
  await navigator.clipboard.writeText(message.content);
  copiedMessageId.value = message.id;
  window.setTimeout(() => {
    if (copiedMessageId.value === message.id) {
      copiedMessageId.value = "";
    }
  }, 1600);
}

async function saveMessageReport(message: ChatMessage) {
  await store.saveAssistantMessageAsReport(message);
}

function formatRole(role: ChatMessageRole) {
  switch (role) {
    case "assistant":
      return "PaperDesk";
    case "system":
      return "系统";
    default:
      return "你";
  }
}

function sessionSummary(session: ChatSession) {
  return session.title || session.last_message_preview || "从这里继续新的知识对话。";
}

function formatAttachmentKind(kind: string) {
  switch (kind) {
    case "image":
      return "图片";
    case "uploaded_pdf":
      return "本地 PDF";
    case "library_document":
      return "库内论文";
    default:
      return kind;
  }
}

function formatDocumentStatus(status: string) {
  switch (status) {
    case "ready":
      return "可用";
    case "processing":
      return "处理中";
    case "failed":
      return "失败";
    default:
      return status;
  }
}

function formatFileKind(kind: string) {
  switch (kind) {
    case "txt":
      return "TXT";
    case "md":
      return "MD";
    case "docx":
      return "DOCX";
    default:
      return "FILE";
  }
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function sessionFileUnavailableReason(file: WorkbenchFileAsset) {
  if (file.failure_reason) {
    return file.failure_reason;
  }
  if (file.status === "unsupported" || file.text_extract_status === "skipped") {
    return "暂未提取文本，可稍后重试或换用 txt / md 文件。";
  }
  if (file.status === "failed" || file.text_extract_status === "failed") {
    return "文件处理失败，请换用 txt / md 文件。";
  }
  if (file.status === "processing" || file.text_extract_status === "pending") {
    return "正在处理，稍后可重新添加。";
  }
  return "当前状态暂不可用于本轮对话。";
}

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

</script>
