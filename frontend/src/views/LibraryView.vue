<template>
  <section class="page-shell content-grid library-layout" @click="handleShellClick">
    <article class="panel">
      <header class="section-head">
        <h2 class="nowrap-title">本地论文库</h2>
        <p>上传 PDF 后，系统会自动解析、建库，并在列表里持续更新处理状态。</p>
      </header>

      <div class="panel-body">
        <label class="upload-box">
          <span>选择 PDF 上传</span>
          <input type="file" accept=".pdf" :disabled="store.submittingUpload" @change="handleUpload" />
        </label>
        <p v-if="store.activeUploadName" class="hint-text">当前文件：{{ store.activeUploadName }}</p>
        <p v-if="store.uploadHint" class="hint-text">{{ store.uploadHint }}</p>
        <p v-if="store.error" class="error-text">{{ store.error }}</p>
        <div
          v-if="store.completionNotice"
          class="floating-notice"
          role="status"
          aria-live="polite"
          @click.stop="store.dismissCompletionNotice"
        >
          <span>{{ store.completionNotice }}</span>
          <button type="button" class="notice-close" @click.stop="store.dismissCompletionNotice">知道了</button>
        </div>
      </div>
    </article>

    <article class="panel">
      <header class="section-head">
        <h2>已上传文档</h2>
        <button class="button-secondary" @click="store.refreshDocuments">刷新</button>
      </header>

      <div class="panel-body panel-scroll">
        <ul class="document-list">
          <li v-for="document in store.documents" :key="document.id" class="document-card">
            <div class="card-copy">
              <div class="task-title-row">
                <strong class="card-title">{{ document.display_name }}</strong>
                <span class="status-badge" :data-status="document.status">
                  {{ formatDocumentStatus(document.status) }}
                </span>
              </div>
              <p class="card-meta">{{ document.title || "暂未提取标题" }}</p>
              <p class="card-meta">{{ document.page_count || 0 }} 页 · {{ formatTime(document.uploaded_at) }}</p>
              <p v-if="document.failure_reason" class="error-text">{{ document.failure_reason }}</p>
            </div>
            <div class="card-actions">
              <button class="button-danger action-button" @click="store.removeDocument(document.id)">
                删除
              </button>
            </div>
          </li>
          <li v-if="!store.documents.length && !store.loading" class="empty-state">
            暂无已上传 PDF。
          </li>
        </ul>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from "vue";

import { useDocumentStore } from "../stores/documents";

const store = useDocumentStore();

onMounted(() => {
  void store.refreshDocuments();
});

async function handleUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  await store.addDocument(file);
  input.value = "";
}

function handleShellClick() {
  if (store.completionNotice) {
    store.dismissCompletionNotice();
  }
}

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN");
}

function formatDocumentStatus(value: string) {
  switch (value) {
    case "processing":
      return "处理中";
    case "ready":
      return "可用";
    case "failed":
      return "处理失败";
    default:
      return value;
  }
}
</script>

<style scoped>
.panel-body {
  position: relative;
}

.floating-notice {
  margin-top: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid rgba(61, 132, 96, 0.18);
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(231, 248, 237, 0.95), rgba(241, 252, 246, 0.98));
  color: #245a3c;
  box-shadow: 0 12px 32px rgba(80, 131, 98, 0.12);
}

.notice-close {
  border: none;
  background: rgba(36, 90, 60, 0.08);
  color: #245a3c;
  padding: 8px 12px;
  border-radius: 999px;
  cursor: pointer;
  font: inherit;
}

.notice-close:hover {
  background: rgba(36, 90, 60, 0.14);
}
</style>
