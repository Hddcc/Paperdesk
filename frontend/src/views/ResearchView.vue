<template>
  <section class="page-shell pdf-workbench">
    <article class="panel pdf-viewer-panel">
      <header class="section-head pdf-viewer-head">
        <div>
          <h2>{{ activeDocument ? documentTitle(activeDocument) : "PDF 阅读" }}</h2>
          <p>{{ activeDocument ? activeDocument.display_name : "从本地论文点击论文名称后，会在这里查看 PDF。" }}</p>
        </div>
      </header>

      <div v-if="openDocuments.length" class="pdf-tab-bar" role="tablist" aria-label="已打开论文">
        <button
          v-for="document in openDocuments"
          :key="document.id"
          class="pdf-tab"
          :class="{ 'pdf-tab-active': document.id === pdfWorkspaceStore.activeDocumentId }"
          type="button"
          role="tab"
          :aria-selected="document.id === pdfWorkspaceStore.activeDocumentId"
          @click="pdfWorkspaceStore.setActiveDocument(document.id)"
        >
          <span>{{ documentTitle(document) }}</span>
          <span
            class="pdf-tab-close"
            role="button"
            tabindex="0"
            aria-label="关闭论文"
            title="关闭论文"
            @click.stop="pdfWorkspaceStore.closeDocument(document.id)"
            @keydown.enter.stop.prevent="pdfWorkspaceStore.closeDocument(document.id)"
            @keydown.space.stop.prevent="pdfWorkspaceStore.closeDocument(document.id)"
          >
            ×
          </span>
        </button>
      </div>

      <div class="panel-body pdf-viewer-body">
        <iframe
          v-for="document in openDocuments"
          v-show="document.id === pdfWorkspaceStore.activeDocumentId"
          :key="document.id"
          class="pdf-frame"
          :src="getDocumentFileUrl(document.id)"
          :title="documentTitle(document)"
        ></iframe>
        <p v-if="!openDocuments.length" class="empty-state pdf-empty">
          暂未选择论文。请回到本地论文页面，点击论文名称打开 PDF。
        </p>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, watch } from "vue";

import { getDocumentFileUrl } from "../services/api";
import { useDocumentStore } from "../stores/documents";
import { usePdfWorkspaceStore } from "../stores/pdfWorkspace";
import type { LibraryDocument } from "../types/models";

defineOptions({
  name: "ResearchView"
});

const documentStore = useDocumentStore();
const pdfWorkspaceStore = usePdfWorkspaceStore();

const openDocuments = computed(() =>
  pdfWorkspaceStore.openDocumentIds
    .map((documentId) => documentStore.documents.find((document) => document.id === documentId))
    .filter((document): document is LibraryDocument => Boolean(document))
);

const activeDocument = computed(() =>
  openDocuments.value.find((document) => document.id === pdfWorkspaceStore.activeDocumentId) ?? null
);

onMounted(() => {
  void documentStore.bootstrapLibrary();
});

watch(
  () => documentStore.documents.map((document) => document.id),
  (documentIds) => {
    pdfWorkspaceStore.reconcileDocuments(documentIds);
  }
);

function documentTitle(document: LibraryDocument) {
  return document.title?.trim() || document.display_name || document.filename;
}
</script>
