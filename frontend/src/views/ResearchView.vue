<template>
  <section class="page-shell research-workbench">
    <div class="research-column research-column-left">
      <article class="panel">
        <header class="section-head">
          <div>
            <h2>研究工作台</h2>
            <p>这里会持续展示主题、参数、任务推进与最终综述。</p>
          </div>
        </header>

        <div class="panel-body status-stack">
          <div class="status-strip">
            <strong>当前状态</strong>
            <span>{{ store.status }}</span>
          </div>

          <div v-if="contextState" class="research-context-strip">
            <span class="status-badge" :data-status="contextState.stage">
              {{ formatContextStage(contextState.stage) }}
            </span>
            <span>{{ contextState.estimated_tokens }} / {{ contextState.budget_tokens }} tokens</span>
            <span>可见步骤 {{ contextState.visible_step_count }}</span>
            <span>压缩证据 {{ contextState.evidence_items_compacted }}</span>
          </div>

          <div v-if="contextState?.sources.length" class="research-context-sources">
            <span
              v-for="source in contextState.sources"
              :key="source"
              class="memory-chip"
            >
              {{ formatContextSource(source) }}
            </span>
          </div>

          <button
            v-if="store.canResumeCurrentRun"
            class="ghost-button"
            type="button"
            :disabled="store.isRunning"
            @click="store.resumeCurrentRun"
          >
            <RefreshCcw :size="16" />
            恢复当前运行
          </button>

          <div v-if="store.topic" class="request-summary">
            <div class="request-summary-row">
              <span class="request-label">研究主题</span>
              <strong>{{ store.topic }}</strong>
            </div>
            <div v-if="store.taskRoute" class="route-summary">
              <span class="status-badge" :data-status="store.taskRoute.evidence_policy">
                {{ formatTaskType(store.taskRoute.task_type) }}
              </span>
              <p>{{ store.taskRoute.rationale }}</p>
              <p class="card-meta">
                产物协议：{{ store.taskRoute.artifact_protocol.title }} ·
                {{ store.taskRoute.artifact_protocol.required_sections.join("、") }}
              </p>
            </div>
            <div class="request-summary-grid">
              <p>
                <span class="request-label">在线论文</span>
                {{ store.lastRequest.top_k_online || 3 }} 篇
              </p>
              <p>
                <span class="request-label">本地证据</span>
                {{ store.lastRequest.top_k_local || 3 }} 条
              </p>
              <p>
                <span class="request-label">检索来源</span>
                {{ formatSearchProvider(store.lastRequest.search_provider || null) }}
              </p>
            </div>
            <p v-if="store.lastRequest.notes" class="task-intent">
              {{ store.lastRequest.notes }}
            </p>
          </div>

          <p v-if="store.error" class="error-text">{{ store.error }}</p>
        </div>
      </article>

      <ResearchLaunchPanel
        compact
        title="补充发起入口"
        description="你也可以直接在工作台里修改参数并重新开始一轮研究。"
        submit-label="开始研究"
        :disabled="store.isRunning"
        :initial-request="launchRequest"
        @submit="handleSubmit"
      />

      <TaskQuickLaunchPanel
        v-if="store.lastRequest.selected_document_ids?.length"
        compact
        title="围绕当前材料继续"
        :document-count="store.lastRequest.selected_document_ids?.length || 0"
        :has-uploaded-context="hasUploadedContext"
        :disabled="store.isRunning"
        @fill="fillQuickAction"
        @submit="submitQuickAction"
      />

      <ResearchTaskList
        :active-task-id="store.activeTaskId"
        :completed-count="store.completedCount"
        :tasks="store.tasks"
        @select="store.setActiveTask"
      />
    </div>

    <TaskSummaryPanel :logs="store.logs" :task-entry="store.activeTask" />

    <article class="panel research-report-panel">
      <header class="section-head">
        <div>
          <h2>最终综述预览</h2>
          <p>研究完成后，可在这里直接查看当前轮次的 Markdown 综述。</p>
        </div>
      </header>
      <div class="panel-body panel-scroll">
        <MarkdownPreview v-if="store.reportMarkdown" :markdown="store.reportMarkdown" />
        <p v-else class="empty-state">综述生成后会显示在这里。</p>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { RefreshCcw } from "lucide-vue-next";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import ResearchLaunchPanel from "../components/ResearchLaunchPanel.vue";
import ResearchTaskList from "../components/ResearchTaskList.vue";
import TaskQuickLaunchPanel, { type QuickLaunchAction } from "../components/TaskQuickLaunchPanel.vue";
import TaskSummaryPanel from "../components/TaskSummaryPanel.vue";
import { useResearchStore } from "../stores/research";
import type { ResearchRequest } from "../types/models";

const store = useResearchStore();
const contextState = computed(() => store.runtimeState?.context_state || null);
const hasUploadedContext = computed(() => store.lastRequest.input_modes?.includes("uploaded_file") ?? false);
const launchRequest = ref<Partial<ResearchRequest>>({ ...store.lastRequest });

onMounted(() => {
  if (store.hasPendingRequest) {
    void store.startPendingResearch();
  }
});

watch(
  () => store.lastRequest,
  (request) => {
    launchRequest.value = { ...request };
  },
  { deep: true }
);

async function handleSubmit(payload: ResearchRequest) {
  await store.startResearch(payload);
}

function fillQuickAction(action: QuickLaunchAction) {
  launchRequest.value = {
    ...store.lastRequest,
    topic: action.prompt,
    notes: action.notes
  };
}

async function submitQuickAction(action: QuickLaunchAction) {
  await store.startResearch({
    ...store.lastRequest,
    topic: action.prompt,
    notes: action.notes
  });
}

function formatSearchProvider(value: string | null) {
  switch (value) {
    case "all":
      return "全部来源";
    case "openalex":
      return "仅 OpenAlex";
    case "arxiv":
      return "仅 arXiv";
    default:
      return "自动选择";
  }
}

function formatTaskType(value: string) {
  const labels: Record<string, string> = {
    qa: "问答",
    paper_summary: "单篇总结",
    multi_paper_review: "多篇综述",
    comparison: "对比分析",
    method_explainer: "方法解释",
    research_brief: "路线建议"
  };
  return labels[value] || value;
}

function formatContextStage(value: string) {
  switch (value) {
    case "evidence_compacted":
      return "证据已压缩";
    case "history_compacted":
      return "历史已摘要";
    case "truncated":
      return "上下文截断";
    default:
      return "上下文正常";
  }
}

function formatContextSource(value: string) {
  const labels: Record<string, string> = {
    research_rules: "研究规则",
    run_goal: "运行目标",
    working_summary: "工作记忆",
    active_task: "当前任务",
    recent_steps: "近期步骤",
    compacted_evidence: "压缩证据",
    completed_task_summaries: "任务摘要"
  };
  return labels[value] || value;
}
</script>
