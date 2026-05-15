import { computed, ref, watch } from "vue";
import { defineStore } from "pinia";

const STORAGE_KEY = "paperdesk-pdf-workspace";

interface PdfWorkspaceState {
  openDocumentIds: string[];
  activeDocumentId: string | null;
}

function loadState(): PdfWorkspaceState {
  if (typeof window === "undefined") {
    return { openDocumentIds: [], activeDocumentId: null };
  }

  try {
    const rawValue = window.localStorage.getItem(STORAGE_KEY);
    if (!rawValue) {
      return { openDocumentIds: [], activeDocumentId: null };
    }
    const parsed = JSON.parse(rawValue) as Partial<PdfWorkspaceState>;
    const openDocumentIds = Array.isArray(parsed.openDocumentIds)
      ? parsed.openDocumentIds.filter((value): value is string => typeof value === "string")
      : [];
    const activeDocumentId =
      typeof parsed.activeDocumentId === "string" && openDocumentIds.includes(parsed.activeDocumentId)
        ? parsed.activeDocumentId
        : openDocumentIds[0] ?? null;
    return { openDocumentIds: [...new Set(openDocumentIds)], activeDocumentId };
  } catch {
    return { openDocumentIds: [], activeDocumentId: null };
  }
}

export const usePdfWorkspaceStore = defineStore("pdfWorkspace", () => {
  const initialState = loadState();
  const openDocumentIds = ref<string[]>(initialState.openDocumentIds);
  const activeDocumentId = ref<string | null>(initialState.activeDocumentId);

  const hasOpenDocuments = computed(() => openDocumentIds.value.length > 0);

  function openDocument(documentId: string) {
    if (!openDocumentIds.value.includes(documentId)) {
      openDocumentIds.value = [...openDocumentIds.value, documentId];
    }
    activeDocumentId.value = documentId;
  }

  function setActiveDocument(documentId: string) {
    if (openDocumentIds.value.includes(documentId)) {
      activeDocumentId.value = documentId;
    }
  }

  function closeDocument(documentId: string) {
    const currentIndex = openDocumentIds.value.indexOf(documentId);
    if (currentIndex === -1) {
      return;
    }

    const nextOpenDocumentIds = openDocumentIds.value.filter((id) => id !== documentId);
    openDocumentIds.value = nextOpenDocumentIds;

    if (activeDocumentId.value !== documentId) {
      return;
    }

    activeDocumentId.value =
      nextOpenDocumentIds[currentIndex] ?? nextOpenDocumentIds[currentIndex - 1] ?? null;
  }

  function reconcileDocuments(existingDocumentIds: string[]) {
    const existingIds = new Set(existingDocumentIds);
    openDocumentIds.value = openDocumentIds.value.filter((documentId) => existingIds.has(documentId));
    if (activeDocumentId.value && !existingIds.has(activeDocumentId.value)) {
      activeDocumentId.value = openDocumentIds.value[0] ?? null;
    }
  }

  watch(
    [openDocumentIds, activeDocumentId],
    () => {
      if (typeof window === "undefined") {
        return;
      }
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          openDocumentIds: openDocumentIds.value,
          activeDocumentId: activeDocumentId.value
        })
      );
    },
    { deep: true }
  );

  return {
    openDocumentIds,
    activeDocumentId,
    hasOpenDocuments,
    openDocument,
    setActiveDocument,
    closeDocument,
    reconcileDocuments
  };
});
