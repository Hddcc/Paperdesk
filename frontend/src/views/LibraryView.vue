<template>
  <section class="page-shell content-grid library-layout">
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
              <p class="card-meta">
                {{ document.title || "暂未提取标题" }}
              </p>
              <p class="card-meta">
                {{ document.page_count || 0 }} 页 · {{ formatTime(document.uploaded_at) }}
              </p>
            </div>
            <div class="card-actions">
              <button
                class="button-danger action-button"
                :disabled="document.status === 'processing'"
                @click="store.removeDocument(document.id)"
              >
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
