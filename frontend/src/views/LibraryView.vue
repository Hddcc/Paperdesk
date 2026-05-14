<template>
  <section class="page-shell library-layout" @click="handleShellClick">
    <div
      v-if="libraryToastMessage"
      class="library-toast"
      :class="{ 'library-toast-error': store.error }"
      role="status"
      aria-live="polite"
      @click.stop
    >
      <div>
        <strong>{{ store.error ? "提示" : "处理完成" }}</strong>
        <p>{{ libraryToastMessage }}</p>
      </div>
      <button type="button" class="notice-close" @click.stop="dismissLibraryToast">
        知道了
      </button>
    </div>

    <section class="library-toolbar" aria-label="论文库操作">
      <div class="library-action-group">
        <label class="button-primary library-file-trigger" :class="{ 'library-file-trigger-disabled': store.submittingUpload }">
          <UploadCloud :size="16" />
          <span>{{ store.submittingUpload ? "上传中..." : "上传论文" }}</span>
          <input type="file" accept=".pdf" :disabled="store.submittingUpload" @change="handleUpload" />
        </label>
        <button class="button-secondary" type="button" @click="openCreateCategoryDialog">
          <Plus :size="16" />
          新建分类
        </button>
      </div>
      <p class="library-toolbar-hint">当前支持 PDF。</p>
    </section>

    <article class="panel library-documents-panel library-table-panel">
      <header class="library-table-head">
        <h2>论文列表</h2>
        <button class="button-secondary" @click="store.refreshDocuments">
          <RefreshCcw :size="16" />
          刷新
        </button>
      </header>

      <div class="library-category-bar">
        <button
          class="category-filter"
          :class="{ 'category-filter-active': store.activeCategoryId === null }"
          type="button"
          @click="store.setActiveCategory(null)"
        >
          全部
          <span>{{ store.documents.length }}</span>
        </button>
        <button
          v-for="category in store.categories"
          :key="category.id"
          class="category-filter"
          :class="{ 'category-filter-active': store.activeCategoryId === category.id }"
          type="button"
          @click="store.setActiveCategory(category.id)"
        >
          <i :style="{ backgroundColor: category.color || '#0f5fb8' }"></i>
          {{ category.name }}
          <span>{{ categoryCount(category.id) }}</span>
        </button>
        <button
          v-if="selectedCategory"
          class="button-danger category-delete-action"
          type="button"
          @click="openDeleteCategoryDialog"
        >
          <Trash2 :size="15" />
          删除分类
        </button>
      </div>

      <div class="panel-body panel-scroll">
        <div class="library-table" role="table" aria-label="论文列表">
          <div class="library-table-row library-table-header" role="row">
            <button
              v-for="column in sortableColumns"
              :key="column.key"
              class="library-sort-button"
              type="button"
              role="columnheader"
              :aria-sort="ariaSort(column.key)"
              @click="toggleSort(column.key)"
            >
              {{ column.label }}
              <span class="sort-indicator" aria-hidden="true">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M5.42955 10.4341C5.74197 10.7466 6.2485 10.7466 6.56092 10.4341L8.62955 8.36551C9.13352 7.86154 8.77659 6.99983 8.06386 6.99983H3.9266C3.21388 6.99983 2.85695 7.86154 3.36092 8.36551L5.42955 10.4341Z"
                    :fill="sortKey === column.key && sortDirection === 'desc' ? '#343A40' : '#B7BBC9'"
                  />
                  <path
                    d="M5.42955 1.56586C5.74197 1.25344 6.2485 1.25344 6.56092 1.56586L8.62955 3.63449C9.13352 4.13846 8.77659 5.00017 8.06386 5.00017H3.9266C3.21388 5.00017 2.85695 4.13846 3.36092 3.63449L5.42955 1.56586Z"
                    :fill="sortKey === column.key && sortDirection === 'asc' ? '#343A40' : '#B7BBC9'"
                  />
                </svg>
              </span>
            </button>
            <span class="library-actions-header" role="columnheader">操作</span>
          </div>

          <div v-for="document in sortedVisibleDocuments" :key="document.id" class="library-table-row" role="row">
            <div class="library-file-cell" role="cell">
              <span class="library-file-icon" :data-status="document.status" aria-hidden="true">PDF</span>
              <div class="library-file-copy">
                <strong>{{ documentTitle(document) }}</strong>
                <small v-if="document.failure_reason">{{ document.failure_reason }}</small>
                <small v-else>{{ document.filename }}</small>
              </div>
            </div>
            <span class="library-muted-cell" role="cell">{{ formatTime(document.uploaded_at) }}</span>
            <span class="library-muted-cell" role="cell">{{ document.page_count || 0 }} 页</span>
            <span role="cell">
              <span class="status-badge" :data-status="document.status">
                {{ formatDocumentStatus(document.status) }}
              </span>
            </span>
            <div class="library-category-cell" role="cell">
              <span
                v-for="category in document.categories"
                :key="category.id"
                class="category-chip"
                :style="{ borderColor: category.color || '#0f5fb8' }"
              >
                {{ category.name }}
              </span>
              <span v-if="!document.categories.length" class="library-muted-cell">无</span>
            </div>
            <div class="library-actions" role="cell">
              <button
                class="icon-button"
                title="分类"
                aria-label="设置论文分类"
                @click="openCategoryDialog(document)"
              >
                <Tags :size="17" />
              </button>
              <button class="icon-danger-button" title="删除" aria-label="删除论文" @click="store.removeDocument(document.id)">
                <Trash2 :size="17" />
              </button>
            </div>
          </div>

          <p v-if="!sortedVisibleDocuments.length && !store.loading" class="empty-state library-empty">
            暂无已上传 PDF。
          </p>
        </div>
      </div>
    </article>

    <div
      v-if="showCreateCategoryDialog"
      class="modal-backdrop"
      role="presentation"
      @click.self="closeCreateCategoryDialog"
    >
      <form class="category-dialog" role="dialog" aria-modal="true" aria-label="新建分类" @submit.prevent="handleCreateCategory">
        <header class="section-head">
          <div>
            <p class="eyebrow">分类</p>
            <h2>新建分类</h2>
            <p>为论文库添加一个新的整理标签。</p>
          </div>
          <button class="icon-button" type="button" aria-label="关闭" @click="closeCreateCategoryDialog">
            ×
          </button>
        </header>

        <label class="category-name-field">
          <span>分类名称</span>
          <input v-model="newCategoryName" maxlength="40" placeholder="例如：中文、综述、待阅读" autofocus />
        </label>

        <footer class="category-dialog-actions">
          <button class="button-secondary" type="button" @click="closeCreateCategoryDialog">取消</button>
          <button class="button-primary" type="submit" :disabled="!newCategoryName.trim()">
            创建分类
          </button>
        </footer>
      </form>
    </div>

    <div
      v-if="categoryToDelete"
      class="modal-backdrop"
      role="presentation"
      @click.self="closeDeleteCategoryDialog"
    >
      <section class="category-dialog" role="dialog" aria-modal="true" aria-label="删除分类">
        <header class="section-head">
          <div>
            <p class="eyebrow">危险操作</p>
            <h2>删除分类</h2>
            <p>确定删除“{{ categoryToDelete.name }}”吗？该分类会从相关论文上移除，论文文件不会被删除。</p>
          </div>
          <button class="icon-button" type="button" aria-label="关闭" @click="closeDeleteCategoryDialog">
            ×
          </button>
        </header>

        <footer class="category-dialog-actions">
          <button class="button-secondary" type="button" @click="closeDeleteCategoryDialog">取消</button>
          <button class="button-danger" type="button" @click="confirmDeleteCategory">
            确认删除
          </button>
        </footer>
      </section>
    </div>

    <div
      v-if="categoryDialogDocument"
      class="modal-backdrop"
      role="presentation"
      @click.self="closeCategoryDialog"
    >
      <section class="category-dialog" role="dialog" aria-modal="true" aria-label="设置论文分类">
        <header class="section-head">
          <div>
            <p class="eyebrow">分类</p>
            <h2>设置论文分类</h2>
            <p>{{ documentTitle(categoryDialogDocument) }}</p>
          </div>
          <button class="icon-button" type="button" aria-label="关闭" @click="closeCategoryDialog">
            ×
          </button>
        </header>

        <div class="category-choice-list">
          <label v-for="category in store.categories" :key="category.id" class="category-choice">
            <input v-model="draftCategoryIds" type="checkbox" :value="category.id" />
            <span :style="{ backgroundColor: category.color || '#0f5fb8' }"></span>
            <strong>{{ category.name }}</strong>
          </label>
          <p v-if="!store.categories.length" class="empty-state">
            暂无分类，请先在论文列表上方新建分类。
          </p>
        </div>

        <footer class="category-dialog-actions">
          <button class="button-secondary" type="button" @click="closeCategoryDialog">取消</button>
          <button class="button-primary" type="button" @click="saveCategoryDialog">
            保存分类
          </button>
        </footer>
      </section>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Plus, RefreshCcw, Tags, Trash2, UploadCloud } from "lucide-vue-next";

import { useDocumentStore } from "../stores/documents";
import type { DocumentCategory, LibraryDocument } from "../types/models";

type SortKey = "title" | "uploaded_at" | "page_count" | "status" | "categories";
type SortDirection = "asc" | "desc";

const store = useDocumentStore();
const newCategoryName = ref("");
const categoryDialogDocument = ref<LibraryDocument | null>(null);
const categoryToDelete = ref<DocumentCategory | null>(null);
const draftCategoryIds = ref<string[]>([]);
const showCreateCategoryDialog = ref(false);
const sortKey = ref<SortKey | null>(null);
const sortDirection = ref<SortDirection>("asc");
let errorToastTimer: number | null = null;
const sortableColumns: Array<{ key: SortKey; label: string }> = [
  { key: "title", label: "文件名" },
  { key: "uploaded_at", label: "上传日期" },
  { key: "page_count", label: "页数" },
  { key: "status", label: "状态" },
  { key: "categories", label: "分类" }
];

const selectedCategory = computed(() =>
  store.categories.find((category) => category.id === store.activeCategoryId) ?? null
);

const libraryToastMessage = computed(() => store.error || store.completionNotice);

const sortedVisibleDocuments = computed(() => {
  if (!sortKey.value) {
    return store.visibleDocuments;
  }

  const direction = sortDirection.value === "asc" ? 1 : -1;
  return [...store.visibleDocuments].sort((left, right) => {
    const result = compareDocuments(left, right, sortKey.value as SortKey);
    return result * direction;
  });
});

onMounted(() => {
  void store.bootstrapLibrary();
});

watch(
  () => store.error,
  (message) => {
    if (errorToastTimer !== null) {
      window.clearTimeout(errorToastTimer);
      errorToastTimer = null;
    }
    if (!message) {
      return;
    }
    errorToastTimer = window.setTimeout(() => {
      store.error = "";
      errorToastTimer = null;
    }, 4000);
  }
);

async function handleCreateCategory() {
  const name = newCategoryName.value.trim();
  if (!name) {
    return;
  }
  await store.addCategory(name);
  newCategoryName.value = "";
  closeCreateCategoryDialog();
}

function openCreateCategoryDialog() {
  newCategoryName.value = "";
  showCreateCategoryDialog.value = true;
}

function closeCreateCategoryDialog() {
  showCreateCategoryDialog.value = false;
  newCategoryName.value = "";
}

function openDeleteCategoryDialog() {
  if (!selectedCategory.value) {
    return;
  }
  categoryToDelete.value = selectedCategory.value;
}

function closeDeleteCategoryDialog() {
  categoryToDelete.value = null;
}

async function confirmDeleteCategory() {
  if (!categoryToDelete.value) {
    return;
  }
  await store.removeCategory(categoryToDelete.value.id);
  closeDeleteCategoryDialog();
}

function toggleSort(key: SortKey) {
  if (sortKey.value === key) {
    sortDirection.value = sortDirection.value === "asc" ? "desc" : "asc";
    return;
  }
  sortKey.value = key;
  sortDirection.value = "asc";
}

function ariaSort(key: SortKey) {
  if (sortKey.value !== key) {
    return "none";
  }
  return sortDirection.value === "asc" ? "ascending" : "descending";
}

function compareDocuments(left: LibraryDocument, right: LibraryDocument, key: SortKey) {
  switch (key) {
    case "title":
      return compareText(documentTitle(left), documentTitle(right));
    case "uploaded_at":
      return new Date(left.uploaded_at).getTime() - new Date(right.uploaded_at).getTime();
    case "page_count":
      return (left.page_count || 0) - (right.page_count || 0);
    case "status":
      return statusRank(left.status) - statusRank(right.status);
    case "categories":
      return compareText(categorySortText(left), categorySortText(right));
    default:
      return 0;
  }
}

function compareText(left: string, right: string) {
  return left.localeCompare(right, "zh-CN", {
    numeric: true,
    sensitivity: "base"
  });
}

function statusRank(status: string) {
  const ranks: Record<string, number> = {
    processing: 0,
    ready: 1,
    failed: 2
  };
  return ranks[status] ?? 3;
}

function categorySortText(document: LibraryDocument) {
  return document.categories.map((category) => category.name).join(" ");
}

function openCategoryDialog(document: LibraryDocument) {
  categoryDialogDocument.value = document;
  draftCategoryIds.value = document.categories.map((category) => category.id);
}

function closeCategoryDialog() {
  categoryDialogDocument.value = null;
  draftCategoryIds.value = [];
}

async function saveCategoryDialog() {
  if (!categoryDialogDocument.value) {
    return;
  }
  await store.saveDocumentCategories(categoryDialogDocument.value.id, draftCategoryIds.value);
  closeCategoryDialog();
}

function categoryCount(categoryId: string) {
  return store.documents.filter((document) =>
    document.categories.some((category) => category.id === categoryId)
  ).length;
}

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

function dismissLibraryToast() {
  store.dismissCompletionNotice();
  store.error = "";
}

function formatTime(value: string) {
  return new Date(value).toLocaleDateString("zh-CN");
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

function documentTitle(document: LibraryDocument) {
  return document.title?.trim() || document.display_name || document.filename;
}
</script>
