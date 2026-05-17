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
          <p>可直接提问，也可以附加图片或论文材料，再生成问答、总结、综述与对比结果。</p>
        </div>
      </header>

      <div ref="messagePanelRef" class="panel-body panel-scroll chat-message-panel">
        <div v-if="store.loading" class="empty-state">正在加载会话…</div>

        <div v-else-if="!store.messages.length" class="chat-empty-state">
          <div class="chat-empty-copy">
            <h3>开始一轮新的知识对话</h3>
            <p>你可以直接提问，也可以先附加图片、刚上传的论文 PDF，或从论文库里选择已入库文档。</p>
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

        <div v-if="showDocumentPicker" class="document-picker-sheet">
          <div class="sheet-head">
            <strong>从论文库选择</strong>
          <button class="button-secondary" @click="showDocumentPicker = false">收起</button>
          </div>
          <div class="document-pick-grid">
            <button
              v-for="document in readyDocuments"
              :key="document.id"
              class="pick-card pick-card-button"
              :class="{ 'task-card-active': store.selectedDocumentIds.includes(document.id) }"
              @click="store.toggleLibraryDocument(document)"
            >
              <strong>{{ document.display_name }}</strong>
              <p class="card-meta">{{ document.title || "未提取标题" }}</p>
              <p class="card-meta">{{ document.page_count || 0 }} 页 · {{ formatDocumentStatus(document.status) }}</p>
            </button>
            <p v-if="!readyDocuments.length" class="empty-state">
              暂无可选择的已入库论文，请先到本地论文库上传并等待处理完成。
            </p>
          </div>
        </div>

        <div v-if="showAttachMenu" class="attach-popover">
          <button class="button-secondary" @click="triggerImageInput">上传图片</button>
          <button class="button-secondary" @click="triggerPdfInput">上传本地 PDF</button>
          <button class="button-secondary" @click="toggleDocumentPicker">选择库内论文</button>
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
          <button class="composer-plus" aria-label="添加附件" @click="showAttachMenu = !showAttachMenu">
            <Paperclip :size="19" />
          </button>
          <textarea
            v-model="store.composerText"
            class="chat-textarea"
            :style="{ height: `${composerHeight}px` }"
            placeholder="输入问题，或附加 PDF / 库内论文后发送。"
            @keydown="handleKeydown"
          />
          <button
            class="button-primary composer-send"
            :disabled="!store.sending && !store.composerText.trim()"
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
      ref="imageInputRef"
      hidden
      type="file"
      accept="image/*"
      @change="handleImageSelected"
    />
    <input
      ref="pdfInputRef"
      hidden
      type="file"
      accept=".pdf"
      @change="handlePdfSelected"
    />
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Copy, FileText, Paperclip, Plus, Save, SendHorizontal, StopCircle, Trash2, X } from "lucide-vue-next";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { useDocumentStore } from "../stores/documents";
import { useKnowledgeStore } from "../stores/knowledge";
import type { ChatAttachment, ChatMessage, ChatMessageRole, ChatSession } from "../types/models";

const store = useKnowledgeStore();
const documentStore = useDocumentStore();

const messagePanelRef = ref<HTMLElement | null>(null);
const imageInputRef = ref<HTMLInputElement | null>(null);
const pdfInputRef = ref<HTMLInputElement | null>(null);
const showAttachMenu = ref(false);
const showDocumentPicker = ref(false);
const copiedMessageId = ref("");
const composerHeight = ref(112);
const resizeStartY = ref(0);
const resizeStartHeight = ref(0);
const resizingComposer = ref(false);

const readyDocuments = computed(() =>
  documentStore.documents.filter((document) => document.status === "ready")
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

function triggerImageInput() {
  showAttachMenu.value = false;
  imageInputRef.value?.click();
}

function triggerPdfInput() {
  showAttachMenu.value = false;
  pdfInputRef.value?.click();
}

function toggleDocumentPicker() {
  showAttachMenu.value = false;
  showDocumentPicker.value = !showDocumentPicker.value;
}

async function handleImageSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  const dataUrl = await readFileAsDataUrl(file);
  const attachment: ChatAttachment = {
    id: crypto.randomUUID(),
    kind: "image",
    display_name: file.name,
    mime_type: file.type,
    data_url: dataUrl,
    status: "ready",
    metadata: {
      size: file.size
    }
  };
  store.queueImageAttachment(attachment);
  input.value = "";
}

async function handlePdfSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  try {
    const document = await documentStore.addDocument(file);
    if (document) {
      store.toggleLibraryDocument(document);
      store.markUploadedTaskDocument(document.id);
    }
  } catch {
    store.queueLocalPdfAttachment(file);
  }
  input.value = "";
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("读取图片失败"));
    reader.readAsDataURL(file);
  });
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!store.sending && store.composerText.trim()) {
      void sendChatMessage();
    }
  }
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
  }
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

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

</script>
