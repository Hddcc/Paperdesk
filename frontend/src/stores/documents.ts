import { defineStore } from "pinia";
import { ref } from "vue";

import { deleteDocument, listDocuments, uploadDocument } from "../services/api";
import type { LibraryDocument } from "../types/models";

export const useDocumentStore = defineStore("documents", () => {
  const documents = ref<LibraryDocument[]>([]);
  const loading = ref(false);
  const error = ref("");
  const uploadHint = ref("");
  const completionNotice = ref("");
  const activeUploadName = ref("");
  const submittingUpload = ref(false);
  const processingUploads = ref<string[]>([]);
  let pollingPromise: Promise<void> | null = null;
  let noticeTimer: number | null = null;

  function syncProcessingUploads() {
    processingUploads.value = documents.value
      .filter((document) => document.status === "processing")
      .map((document) => document.id);
  }

  function showCompletionNotice(message: string) {
    completionNotice.value = message;
    if (noticeTimer !== null) {
      window.clearTimeout(noticeTimer);
    }
    noticeTimer = window.setTimeout(() => {
      completionNotice.value = "";
      noticeTimer = null;
    }, 4000);
  }

  function dismissCompletionNotice() {
    completionNotice.value = "";
    if (noticeTimer !== null) {
      window.clearTimeout(noticeTimer);
      noticeTimer = null;
    }
  }

  function clearProcessingHintIfIdle() {
    if (!processingUploads.value.length && !documents.value.some((document) => document.status === "failed")) {
      uploadHint.value = "";
    }
  }

  async function fetchDocuments(options: { background?: boolean } = {}) {
    if (!options.background) {
      loading.value = true;
      error.value = "";
    }
    try {
      documents.value = await listDocuments();
      syncProcessingUploads();
      clearProcessingHintIfIdle();
      if (processingUploads.value.length) {
        void ensureProcessingPoll();
      }
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

  async function addDocument(file: File): Promise<LibraryDocument | undefined> {
    submittingUpload.value = true;
    activeUploadName.value = file.name;
    uploadHint.value = "正在上传，请稍候。";
    error.value = "";
    try {
      const document = await uploadDocument(file);
      upsertDocument(document);
      if (document.status === "processing") {
        uploadHint.value = "文件已上传，系统正在后台解析并建库，列表会自动刷新。";
        trackProcessingDocument(document.id);
        void ensureProcessingPoll();
      } else if (document.status === "ready") {
        uploadHint.value = "";
        showCompletionNotice("PDF 已处理完成，现在可以直接使用。");
      } else if (document.status === "failed") {
        uploadHint.value = document.failure_reason
          ? `上传完成，但处理失败：${document.failure_reason}`
          : "上传完成，但文档处理失败。";
      } else {
        uploadHint.value = `上传已接收，当前状态：${document.status}`;
      }
      return document;
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
      processingUploads.value = processingUploads.value.filter((item) => item !== documentId);
      clearProcessingHintIfIdle();
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
    } else {
      const next = [...documents.value];
      next[index] = document;
      documents.value = next;
    }
    syncProcessingUploads();
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
          const failedDocuments = documents.value.filter((document) => document.status === "failed");
          const latestReason = failedDocuments[0]?.failure_reason;
          uploadHint.value = latestReason
            ? `${completedFailed} 篇文档处理失败：${latestReason}`
            : `${completedFailed} 篇文档处理失败，请刷新列表查看状态。`;
        } else if (completedReady > 0) {
          uploadHint.value = "";
          showCompletionNotice(
            completedReady === 1
              ? "PDF 已处理完成，现在可以直接使用。"
              : `${completedReady} 篇 PDF 已处理完成，现在可以直接使用。`
          );
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
    completionNotice,
    activeUploadName,
    submittingUpload,
    refreshDocuments,
    addDocument,
    removeDocument,
    dismissCompletionNotice
  };
});
