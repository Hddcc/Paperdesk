import { defineStore } from "pinia";
import { ref } from "vue";

import { deleteDocument, listDocuments, uploadDocument } from "../services/api";
import type { LibraryDocument } from "../types/models";

export const useDocumentStore = defineStore("documents", () => {
  const documents = ref<LibraryDocument[]>([]);
  const loading = ref(false);
  const error = ref("");

  async function refreshDocuments() {
    loading.value = true;
    error.value = "";
    try {
      documents.value = await listDocuments();
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载文档失败";
    } finally {
      loading.value = false;
    }
  }

  async function addDocument(file: File) {
    loading.value = true;
    error.value = "";
    try {
      const document = await uploadDocument(file);
      documents.value = [document, ...documents.value];
    } catch (err) {
      error.value = err instanceof Error ? err.message : "上传文档失败";
      throw err;
    } finally {
      loading.value = false;
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

  return {
    documents,
    loading,
    error,
    refreshDocuments,
    addDocument,
    removeDocument
  };
});

