<template>
  <article class="panel">
    <header class="section-head">
      <div>
        <h2>任务总结</h2>
        <p>查看当前任务的阶段性结论、证据来源和过程记录。</p>
      </div>
    </header>

    <div class="panel-body panel-scroll task-detail-panel">
      <template v-if="taskEntry">
        <section class="task-detail-section">
          <div class="task-title-row">
            <div>
              <h3>{{ taskEntry.task.title }}</h3>
              <p class="task-intent">{{ taskEntry.task.intent }}</p>
            </div>
            <span class="status-badge" :data-status="taskEntry.task.status">
              {{ formatTaskStatus(taskEntry.task.status) }}
            </span>
          </div>
        </section>

        <section class="task-detail-section">
          <h4>总结内容</h4>
          <MarkdownPreview
            v-if="summaryMarkdown"
            class="summary-markdown"
            :markdown="summaryMarkdown"
          />
          <p v-else class="empty-state">当前任务还在处理中，任务总结返回后会显示在这里。</p>
        </section>

        <section class="task-detail-section">
          <h4>关联证据</h4>
          <ul v-if="evidenceItems.length" class="evidence-list">
            <li v-for="item in evidenceItems" :key="item.id" class="evidence-card">
              <div class="evidence-head">
                <strong>{{ item.title || item.citation_label }}</strong>
                <span class="status-badge evidence-source">
                  {{ formatSourceType(item.source_type) }}
                </span>
              </div>
              <p class="task-intent">
                {{ item.quote || item.snippet || "该证据暂未返回摘录。" }}
              </p>
              <p class="evidence-meta">
                {{ item.citation_label }}
                <span v-if="item.page_number"> · 第 {{ item.page_number }} 页</span>
              </p>
              <a v-if="item.url" :href="item.url" class="inline-link" target="_blank" rel="noreferrer">
                查看来源
              </a>
            </li>
          </ul>
          <p v-else class="empty-state">当前任务暂未返回可展示的证据。</p>
        </section>

        <section class="task-detail-section">
          <h4>相关论文</h4>
          <ul v-if="paperRecords.length" class="paper-list">
            <li v-for="paper in paperRecords" :key="paper.paper_id || paper.title" class="paper-card">
              <strong>{{ paper.title }}</strong>
              <p class="task-intent">
                {{ paper.authors.join("、") || "作者信息待补充" }}
                <span v-if="paper.year"> · {{ paper.year }}</span>
                <span v-if="paper.venue"> · {{ paper.venue }}</span>
              </p>
            </li>
          </ul>
          <p v-else class="empty-state">当前任务暂未整理出独立论文列表。</p>
        </section>
      </template>

      <section class="task-detail-section">
        <h4>过程记录</h4>
        <ol v-if="recentLogs.length" class="log-list">
          <li v-for="(log, index) in recentLogs" :key="`${index}-${log}`">{{ log }}</li>
        </ol>
        <p v-else class="empty-state">研究开始后，这里会显示状态推进记录。</p>
      </section>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed } from "vue";

import MarkdownPreview from "./MarkdownPreview.vue";
import type { EvidenceItem, PaperRecord, ResearchTaskState } from "../types/models";

const props = defineProps<{
  taskEntry: ResearchTaskState | null;
  logs: string[];
}>();

const summaryMarkdown = computed(
  () => props.taskEntry?.summary?.summary_markdown || props.taskEntry?.task.summary_markdown || ""
);
const evidenceItems = computed<EvidenceItem[]>(
  () => props.taskEntry?.summary?.evidence_items || props.taskEntry?.evidenceItems || []
);
const paperRecords = computed<PaperRecord[]>(
  () => props.taskEntry?.summary?.paper_records || props.taskEntry?.papers || []
);
const recentLogs = computed(() => props.logs.slice(-8).reverse());

function formatTaskStatus(value: string) {
  switch (value) {
    case "pending":
      return "待处理";
    case "in_progress":
      return "进行中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    default:
      return value;
  }
}

function formatSourceType(value: string) {
  return value === "local_document" ? "本地文档" : "在线论文";
}
</script>
