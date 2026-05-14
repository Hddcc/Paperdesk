<template>
  <section class="page-shell content-grid reports-layout">
    <article class="panel">
      <header class="section-head">
        <h2>历史报告</h2>
        <button class="button-secondary" @click="store.refreshReports">
          <RefreshCcw :size="16" />
          刷新
        </button>
      </header>
      <div class="panel-body panel-scroll">
        <ul class="report-list">
          <li
            v-for="report in store.reports"
            :key="report.id"
            class="report-card"
            :class="{ 'report-card-active': report.id === store.activeReport?.id }"
          >
            <div class="report-card-row">
              <button class="report-link report-card-main" @click="store.loadReport(report.id)">
                <strong class="card-title">{{ report.topic }}</strong>
                <span class="card-meta">{{ formatTime(report.created_at) }}</span>
              </button>
              <div class="card-actions">
                <button class="button-danger action-button" @click="store.removeReport(report.id)">
                  <Trash2 :size="15" />
                  删除
                </button>
              </div>
            </div>
          </li>
          <li v-if="!store.reports.length && !store.loading" class="empty-state">
            暂无报告，请先在研究工作台运行一次研究。
          </li>
        </ul>
        <p v-if="store.error" class="error-text">{{ store.error }}</p>
      </div>
    </article>

    <article class="panel report-preview-panel">
      <header class="section-head">
        <div>
          <h2>报告预览</h2>
          <p>聚焦阅读最终综述，并在需要时导出 Markdown。</p>
        </div>
        <button
          class="button-secondary"
          :disabled="!store.activeReport || exporting"
          @click="handleExport"
        >
          <Download :size="16" />
          {{ exporting ? "导出中..." : "导出 Markdown" }}
        </button>
      </header>
      <div class="panel-body panel-scroll">
        <MarkdownPreview v-if="store.activeReport" :markdown="store.activeReport.markdown" />
        <p v-else class="empty-state">请选择左侧的一份报告。</p>
        <p v-if="exportError" class="error-text">{{ exportError }}</p>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { Download, RefreshCcw, Trash2 } from "lucide-vue-next";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { exportReportMarkdown } from "../services/api";
import { useReportStore } from "../stores/reports";

const store = useReportStore();
const exporting = ref(false);
const exportError = ref("");

onMounted(() => {
  void hydrateReports();
});

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN");
}

async function hydrateReports() {
  await store.refreshReports();
  if (!store.activeReport && store.reports.length) {
    await store.loadReport(store.reports[0].id);
  }
}

async function handleExport() {
  if (!store.activeReport) {
    return;
  }

  exporting.value = true;
  exportError.value = "";
  try {
    const markdown = await exportReportMarkdown(store.activeReport.id);
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const objectUrl = window.URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = `${store.activeReport.id}.md`;
    anchor.click();
    window.URL.revokeObjectURL(objectUrl);
  } catch (err) {
    exportError.value = err instanceof Error ? err.message : "导出失败";
  } finally {
    exporting.value = false;
  }
}
</script>
