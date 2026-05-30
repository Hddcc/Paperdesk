import { defineStore } from "pinia";
import { computed, ref } from "vue";

import {
  assignDocumentCategories,
  createDocumentCategory,
  deleteDocument,
  deleteDocumentCategory,
  listDocumentCategories,
  listDocuments,
  uploadDocument
} from "../services/api";
import type { DocumentCategory, LibraryDocument } from "../types/models";

export const useDocumentStore = defineStore("documents", () => {
  const documents = ref<LibraryDocument[]>([]);
  const categories = ref<DocumentCategory[]>([]);
  const activeCategoryId = ref<string | null>(null);
  const loading = ref(false);
  const categoryLoading = ref(false);
  const error = ref("");
  const uploadHint = ref("");
  const completionNotice = ref("");
  const activeUploadName = ref("");
  const submittingUpload = ref(false);
  const processingUploads = ref<string[]>([]);
  let pollingPromise: Promise<void> | null = null;
  let noticeTimer: number | null = null;

  const visibleDocuments = computed(() => {
    if (!activeCategoryId.value) {
      return documents.value;
    }
    return documents.value.filter((document) =>
      document.categories?.some((category) => category.id === activeCategoryId.value)
    );
  });

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

  async function refreshCategories() {
    categoryLoading.value = true;
    error.value = "";
    try {
      categories.value = await listDocumentCategories();
      if (
        activeCategoryId.value &&
        !categories.value.some((category) => category.id === activeCategoryId.value)
      ) {
        activeCategoryId.value = null;
      }
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载分类失败";
    } finally {
      categoryLoading.value = false;
    }
  }

  async function bootstrapLibrary() {
    await Promise.all([refreshDocuments(), refreshCategories()]);
  }

  function setActiveCategory(categoryId: string | null) {
    activeCategoryId.value = categoryId;
  }

  async function addCategory(name: string) {
    const trimmed = name.trim();
    if (!trimmed) {
      return;
    }
    error.value = "";
    const palette = ["#0f5fb8", "#047c71", "#6957d8", "#b76a00", "#b42318"];
    try {
      const category = await createDocumentCategory({
        name: trimmed,
        color: palette[categories.value.length % palette.length]
      });
      categories.value = [...categories.value, category];
      activeCategoryId.value = category.id;
    } catch (err) {
      error.value = err instanceof Error ? err.message : "创建分类失败";
      throw err;
    }
  }

  async function removeCategory(categoryId: string) {
    const previousCategories = categories.value;
    const previousDocuments = documents.value;
    categories.value = categories.value.filter((category) => category.id !== categoryId);
    documents.value = documents.value.map((document) => ({
      ...document,
      categories: document.categories.filter((category) => category.id !== categoryId)
    }));
    if (activeCategoryId.value === categoryId) {
      activeCategoryId.value = null;
    }
    error.value = "";
    try {
      await deleteDocumentCategory(categoryId);
    } catch (err) {
      categories.value = previousCategories;
      documents.value = previousDocuments;
      error.value = err instanceof Error ? err.message : "删除分类失败";
      throw err;
    }
  }

  async function saveDocumentCategories(
    documentId: string,
    categoryIds: string[],
    options: { confirmClear?: boolean } = {}
  ) {
    const previousDocuments = documents.value;
    const selectedCategories = categories.value.filter((category) => categoryIds.includes(category.id));
    documents.value = documents.value.map((document) =>
      document.id === documentId
        ? {
            ...document,
            categories: selectedCategories
          }
        : document
    );
    error.value = "";
    try {
      const updated = await assignDocumentCategories(documentId, {
        category_ids: categoryIds,
        confirm_clear: options.confirmClear ?? false
      });
      upsertDocument(updated);
    } catch (err) {
      documents.value = previousDocuments;
      error.value = err instanceof Error ? err.message : "保存论文分类失败";
      throw err;
    }
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

  async function addDocuments(files: File[]): Promise<LibraryDocument[]> {
    if (!files.length) {
      return [];
    }
    submittingUpload.value = true;
    error.value = "";
    const uploaded: LibraryDocument[] = [];
    try {
      for (const [index, file] of files.entries()) {
        activeUploadName.value = file.name;
        uploadHint.value =
          files.length === 1
            ? "正在上传，请稍候。"
            : `正在上传 ${index + 1}/${files.length}：${file.name}`;
        const document = await uploadDocument(file);
        uploaded.push(document);
        upsertDocument(document);
        if (document.status === "processing") {
          trackProcessingDocument(document.id);
        }
      }
      if (processingUploads.value.length) {
        uploadHint.value = `${processingUploads.value.length} 篇文档正在处理，列表会自动刷新。`;
        void ensureProcessingPoll();
      } else if (uploaded.some((document) => document.status === "ready")) {
        uploadHint.value = "";
        showCompletionNotice(
          uploaded.length === 1
            ? "PDF 已处理完成，现在可以直接使用。"
            : `${uploaded.length} 篇 PDF 已处理完成，现在可以直接使用。`
        );
      }
      return uploaded;
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
    const previousDocuments = documents.value;
    const previousProcessingUploads = processingUploads.value;
    const targetDocument = documents.value.find((item) => item.id === documentId);

    documents.value = documents.value.filter((item) => item.id !== documentId);
    processingUploads.value = processingUploads.value.filter((item) => item !== documentId);
    error.value = "";
    try {
      await deleteDocument(documentId);
      clearProcessingHintIfIdle();
    } catch (err) {
      error.value = err instanceof Error ? err.message : "删除文档失败";
      if (targetDocument) {
        documents.value = previousDocuments;
        processingUploads.value = previousProcessingUploads;
      }
      throw err;
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
    visibleDocuments,
    categories,
    activeCategoryId,
    loading,
    categoryLoading,
    error,
    uploadHint,
    completionNotice,
    activeUploadName,
    submittingUpload,
    bootstrapLibrary,
    refreshDocuments,
    refreshCategories,
    addDocument,
    addDocuments,
    removeDocument,
    addCategory,
    removeCategory,
    saveDocumentCategories,
    setActiveCategory,
    dismissCompletionNotice
  };
});
