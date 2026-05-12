<template>
  <section class="page-shell knowledge-chat-page">
    <aside class="chat-session-panel panel">
      <header class="chat-panel-head">
        <div>
          <p class="eyebrow">Knowledge Chat</p>
          <h2>知识库对话</h2>
        </div>
        <button class="button-secondary" :disabled="store.loading" @click="store.createNewSession">
          新建对话
        </button>
      </header>

      <div class="panel-body panel-scroll">
        <ul class="chat-session-list">
          <li v-for="session in store.sessions" :key="session.id">
            <button
              class="chat-session-button"
              :class="{ 'chat-session-active': session.id === store.currentSessionId }"
              @click="store.openSession(session.id)"
            >
              <strong>{{ session.title }}</strong>
              <p>{{ session.last_message_preview || "从这里继续新的知识对话。" }}</p>
              <span>{{ formatTime(session.updated_at) }}</span>
            </button>
          </li>
        </ul>
      </div>
    </aside>

    <article class="panel chat-stage">
      <header class="chat-stage-head">
        <div>
          <p class="eyebrow">PaperDesk Agent</p>
          <h2>{{ store.currentSession?.title || "新对话" }}</h2>
          <p>
            直接提问，或通过 “+” 添加图片、本地 PDF、库内论文作为上下文。
          </p>
        </div>
      </header>

      <div v-if="store.memorySnapshot?.items.length" class="memory-strip">
        <span class="memory-label">记忆</span>
        <div class="memory-chip-list">
          <span v-for="item in store.memorySnapshot.items" :key="item.id" class="memory-chip">
            {{ item.summary }}
          </span>
        </div>
      </div>

      <div v-if="store.contextState" class="memory-strip">
        <span class="memory-label">上下文</span>
        <div class="memory-chip-list">
          <span class="memory-chip">
            {{ formatContextStage(store.contextState.stage) }}
          </span>
          <span class="memory-chip">
            {{ store.contextState.estimated_tokens }} / {{ store.contextState.budget_tokens }} tokens
          </span>
          <span
            v-for="source in store.contextState.sources"
            :key="source"
            class="memory-chip"
          >
            {{ formatContextSource(source) }}
          </span>
        </div>
      </div>

      <div class="panel-body panel-scroll chat-message-panel">
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
                v-if="message.role === 'assistant'"
                :markdown="message.content"
              />
              <p v-else class="chat-message-text">{{ message.content }}</p>

              <p v-if="message.warning" class="hint-text">{{ message.warning }}</p>

              <div v-if="message.citations.length" class="citation-strip">
                <span class="request-label">引用</span>
                <div class="memory-chip-list">
                  <span v-for="citation in message.citations" :key="citation" class="citation-chip">
                    {{ citation }}
                  </span>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>

      <div class="chat-composer">
        <div v-if="store.retrievalNotice" class="hint-text">{{ store.retrievalNotice }}</div>
        <p v-if="store.error" class="error-text">{{ store.error }}</p>

        <div v-if="store.draftAttachments.length" class="draft-strip">
          <div class="memory-chip-list">
            <button
              v-for="attachment in store.draftAttachments"
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
          </div>
        </div>

        <div v-if="showAttachMenu" class="attach-popover">
          <button class="button-secondary" @click="triggerImageInput">上传图片</button>
          <button class="button-secondary" @click="triggerPdfInput">上传本地 PDF</button>
          <button class="button-secondary" @click="toggleDocumentPicker">选择库内论文</button>
        </div>

        <div class="composer-shell">
          <button class="composer-plus" @click="showAttachMenu = !showAttachMenu">+</button>
          <textarea
            v-model="store.composerText"
            class="chat-textarea"
            placeholder="输入你的问题，或附加图片 / PDF / 论文库文档后一起发送。"
            @keydown="handleKeydown"
          />
          <button
            class="button-primary composer-send"
            :disabled="store.sending || !store.composerText.trim()"
            @click="store.sendCurrentMessage"
          >
            {{ store.sending ? "发送中..." : "发送" }}
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
import { computed, onMounted, ref } from "vue";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { useDocumentStore } from "../stores/documents";
import { useKnowledgeStore } from "../stores/knowledge";
import type { ChatAttachment, ChatMessageRole, LibraryDocument } from "../types/models";

const store = useKnowledgeStore();
const documentStore = useDocumentStore();

const imageInputRef = ref<HTMLInputElement | null>(null);
const pdfInputRef = ref<HTMLInputElement | null>(null);
const showAttachMenu = ref(false);
const showDocumentPicker = ref(false);

const readyDocuments = computed(() =>
  documentStore.documents.filter((document) => document.status === "ready")
);

onMounted(async () => {
  await Promise.all([documentStore.refreshDocuments(), store.bootstrap()]);
});

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
  const document = await documentStore.addDocument(file);
  if (document) {
    store.attachUploadedDocument(document);
    await documentStore.refreshDocuments();
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
      void store.sendCurrentMessage();
    }
  }
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

function formatContextStage(stage: string) {
  switch (stage) {
    case "evidence_compacted":
      return "证据已轻压缩";
    case "history_compacted":
      return "历史已摘要";
    case "truncated":
      return "已强制截断";
    default:
      return "正常上下文";
  }
}

function formatContextSource(source: string) {
  switch (source) {
    case "system_instruction":
      return "系统指令";
    case "project_rules":
      return "项目规则";
    case "user_preferences":
      return "用户偏好";
    case "session_summary":
      return "会话摘要";
    case "compact_summary":
      return "压缩摘要";
    case "recent_messages":
      return "最近消息";
    case "attachments":
      return "本轮附件";
    case "rag_evidence":
      return "RAG 证据";
    default:
      return source;
  }
}
</script>
