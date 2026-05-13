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
              :title="session.title"
            >
              <strong class="chat-session-title">{{ session.title }}</strong>
              <p>{{ session.last_message_preview || "从这里继续新的知识对话。" }}</p>
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
          <p>直接提问、上传 PDF，或选择库内论文生成问答、总结、综述与对比结果。</p>
        </div>
      </header>

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
            placeholder="输入问题，或附加 PDF / 库内论文后发送或生成结果。"
            @keydown="handleKeydown"
          />
          <button
            class="button-primary composer-send"
            :disabled="store.sending || researchStore.isRunning || !store.composerText.trim()"
            @click="store.sendCurrentMessage"
          >
            {{ store.sending ? "发送中..." : "发送" }}
          </button>
          <button
            class="button-secondary composer-research"
            :disabled="store.sending || researchStore.isRunning || !store.composerText.trim()"
            @click="startResearchTask"
          >
            {{ researchStore.isRunning ? "研究中..." : "生成结果" }}
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
import { useRouter } from "vue-router";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { useDocumentStore } from "../stores/documents";
import { useKnowledgeStore } from "../stores/knowledge";
import { useResearchStore } from "../stores/research";
import type { ChatAttachment, ChatMessageRole, LibraryDocument, ResearchInputMode } from "../types/models";

const store = useKnowledgeStore();
const documentStore = useDocumentStore();
const researchStore = useResearchStore();
const router = useRouter();

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

async function startResearchTask() {
  const content = store.composerText.trim();
  if (!content) {
    return;
  }
  const selectedIds = [...store.selectedDocumentIds];
  const inputModes: ResearchInputMode[] = ["prompt"];
  if (selectedIds.length) {
    inputModes.push("knowledge_base");
  }
  if (store.uploadedTaskDocumentIds.some((documentId) => selectedIds.includes(documentId))) {
    inputModes.push("uploaded_file");
  }
  researchStore.queueResearch({
    topic: content,
    top_k_local: Math.max(3, selectedIds.length || 3),
    top_k_online: 3,
    search_provider: null,
    notes: selectedIds.length ? "从知识库入口发起，优先使用已选择的库内论文。" : "从知识库入口发起。",
    input_modes: inputModes,
    selected_document_ids: selectedIds
  });
  store.clearComposer();
  await router.push("/research");
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

</script>
