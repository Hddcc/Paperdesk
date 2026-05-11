import { defineStore } from "pinia";
import { ref } from "vue";

import { deleteDocument, listDocuments, uploadDocument } from "../services/api";
import type { LibraryDocument } from "../types/models";

export const useDocumentStore = defineStore("documents", () => {
  const documents = ref<LibraryDocument[]>([]);
  const loading = ref(false);
  const error = ref("");
  const uploadHint = ref("");
  const activeUploadName = ref("");
  const submittingUpload = ref(false);
  const processingUploads = ref<string[]>([]);
  let pollingPromise: Promise<void> | null = null;

  async function fetchDocuments(options: { background?: boolean } = {}) {
    if (!options.background) {
      loading.value = true;
      error.value = "";
    }
    try {
      documents.value = await listDocuments();
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载文档失败";
    } finally {
      if (!options.background) {
        loading.value = false;
      }
    }
  }

  async function refreshDocuments() {
    await fetchDocuments();
  }

  async function addDocument(file: File) {
    submittingUpload.value = true;
    activeUploadName.value = file.name;
    uploadHint.value = "正在上传，请稍候。";
    error.value = "";
    try {
      const document = await uploadDocument(file);
      upsertDocument(document);
      if (document.status === "processing") {
        uploadHint.value = "文件已上传，正在处理中，列表会自动更新。";
        trackProcessingDocument(document.id);
        void ensureProcessingPoll();
      } else if (document.status === "ready") {
        uploadHint.value = "上传完成，可以开始使用。";
      } else if (document.status === "failed") {
        uploadHint.value = "上传成功，但处理失败，请刷新后查看状态。";
      } else {
        uploadHint.value = `上传已接收，当前状态：${document.status}`;
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : "上传文档失败";
      uploadHint.value = "";
      throw err;
    } finally {
      submittingUpload.value = false;
      activeUploadName.value = "";
    }
  }

  async function removeDocument(documentId: string) {
    loading.value = true;
    error.value = "";
    try {
      await deleteDocument(documentId);
      documents.value = documents.value.filter((item) => item.id !== documentId);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "删除文档失败";
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function upsertDocument(document: LibraryDocument) {
    const index = documents.value.findIndex((item) => item.id === document.id);
    if (index === -1) {
      documents.value = [document, ...documents.value];
      return;
    }
    const next = [...documents.value];
    next[index] = document;
    documents.value = next;
  }

  function trackProcessingDocument(documentId: string) {
    if (!processingUploads.value.includes(documentId)) {
      processingUploads.value = [...processingUploads.value, documentId];
    }
  }

  async function ensureProcessingPoll() {
    if (pollingPromise) {
      return pollingPromise;
    }

    pollingPromise = (async () => {
      while (processingUploads.value.length) {
        await new Promise((resolve) => window.setTimeout(resolve, 1500));
        await fetchDocuments({ background: true });

        const pending = new Set(processingUploads.value);
        const nextPending: string[] = [];
        let completedReady = 0;
        let completedFailed = 0;

        for (const document of documents.value) {
          if (!pending.has(document.id)) {
            continue;
          }
          if (document.status === "processing") {
            nextPending.push(document.id);
            continue;
          }
          if (document.status === "ready") {
            completedReady += 1;
            continue;
          }
          if (document.status === "failed") {
            completedFailed += 1;
          }
        }

        processingUploads.value = nextPending;
        if (nextPending.length) {
          uploadHint.value = `还有 ${nextPending.length} 篇文档正在处理中。`;
        } else if (completedFailed > 0) {
          uploadHint.value = `${completedFailed} 篇文档处理失败，请刷新列表查看状态。`;
        } else if (completedReady > 0) {
          uploadHint.value = "处理完成，文档已可使用。";
        }
      }
    })().finally(() => {
      pollingPromise = null;
    });

    return pollingPromise;
  }

  return {
    documents,
    loading,
    error,
    uploadHint,
    activeUploadName,
    submittingUpload,
    refreshDocuments,
    addDocument,
    removeDocument
  };
});
