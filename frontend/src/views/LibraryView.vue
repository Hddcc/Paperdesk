<template>
  <section class="content-grid">
    <article class="panel">
      <header class="section-head">
        <h2>本地论文库</h2>
        <p>上传 PDF 后会先接收文件，再在后台完成解析、切片、向量化并接入本地语义检索。</p>
      </header>

      <label class="upload-box">
        <span>选择 PDF 上传</span>
        <input type="file" accept=".pdf" :disabled="store.submittingUpload" @change="handleUpload" />
      </label>
      <p v-if="store.activeUploadName" class="hint-text">当前文件：{{ store.activeUploadName }}</p>
      <p v-if="store.uploadHint" class="hint-text">{{ store.uploadHint }}</p>
      <p v-if="store.error" class="error-text">{{ store.error }}</p>
    </article>

    <article class="panel">
      <header class="section-head">
        <h2>已上传文档</h2>
        <button class="button-secondary" @click="store.refreshDocuments">刷新</button>
      </header>

      <ul class="document-list">
        <li v-for="document in store.documents" :key="document.id" class="document-card">
          <div>
            <strong>{{ document.display_name }}</strong>
            <p>{{ formatDocumentStatus(document.status) }} · {{ document.page_count || 0 }} 页 · {{ formatTime(document.uploaded_at) }}</p>
          </div>
          <button class="button-danger" :disabled="document.status === 'processing'" @click="store.removeDocument(document.id)">
            删除
          </button>
        </li>
        <li v-if="!store.documents.length && !store.loading" class="empty-state">
          暂无已上传 PDF。
        </li>
      </ul>
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
      return "processing（后台处理中）";
    case "ready":
      return "ready";
    case "failed":
      return "failed";
    default:
      return value;
  }
}
</script>
