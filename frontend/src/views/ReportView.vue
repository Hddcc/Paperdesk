<template>
  <section class="content-grid reports-grid">
    <article class="panel">
      <header class="section-head">
        <h2>历史报告</h2>
        <button class="button-secondary" @click="store.refreshReports">刷新</button>
      </header>
      <ul class="report-list">
        <li v-for="report in store.reports" :key="report.id" class="report-card">
          <button class="report-link" @click="store.loadReport(report.id)">
            <strong>{{ report.topic }}</strong>
            <span>{{ formatTime(report.created_at) }}</span>
          </button>
        </li>
        <li v-if="!store.reports.length && !store.loading" class="empty-state">
          暂无报告，请先在研究工作台运行一次研究。
        </li>
      </ul>
      <p v-if="store.error" class="error-text">{{ store.error }}</p>
    </article>

    <article class="panel report-preview-panel">
      <header class="section-head">
        <h2>报告预览</h2>
        <p>读取 SQLite 记录中的 Markdown 报告。</p>
      </header>
      <MarkdownPreview v-if="store.activeReport" :markdown="store.activeReport.markdown" />
      <p v-else class="empty-state">请选择左侧的一份报告。</p>
    </article>
  </section>
</template>

<script setup lang="ts">
import { onMounted } from "vue";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { useReportStore } from "../stores/reports";

const store = useReportStore();

onMounted(() => {
  void store.refreshReports();
});

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN");
}
</script>

