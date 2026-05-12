import { defineStore } from "pinia";
import { computed, ref } from "vue";

import { resumeResearchStream, runResearchStream } from "../services/api";
import type {
  EvidenceItem,
  PaperRecord,
  ResearchReport,
  ResearchRequest,
  ResearchRun,
  ResearchRuntimeState,
  ResearchStreamEvent,
  ResearchTaskState,
  TaskSummary,
  TodoTask
} from "../types/models";

const DEFAULT_REQUEST: ResearchRequest = {
  topic: "",
  top_k_online: 3,
  top_k_local: 3,
  search_provider: null,
  notes: null
};

export const useResearchStore = defineStore("research", () => {
  const run = ref<ResearchRun | null>(null);
  const topic = ref("");
  const lastRequest = ref<ResearchRequest>({ ...DEFAULT_REQUEST });
  const pendingRequest = ref<ResearchRequest | null>(null);
  const isRunning = ref(false);
  const error = ref("");
  const status = ref("尚未开始");
  const logs = ref<string[]>([]);
  const tasks = ref<ResearchTaskState[]>([]);
  const activeTaskId = ref("");
  const finalReport = ref<ResearchReport | null>(null);
  const runtimeState = ref<ResearchRuntimeState | null>(null);

  const completedCount = computed(
    () => tasks.value.filter((entry) => entry.task.status === "completed").length
  );
  const canResumeCurrentRun = computed(
    () =>
      !!run.value &&
      !isRunning.value &&
      run.value.status !== "completed"
  );

  const taskSummaryMap = computed<Record<string, TaskSummary>>(() =>
    Object.fromEntries(
      tasks.value
        .filter((entry): entry is ResearchTaskState & { summary: TaskSummary } => entry.summary !== null)
        .map((entry) => [entry.task.id, entry.summary])
    )
  );

  const activeTask = computed<ResearchTaskState | null>(() => {
    if (!tasks.value.length) {
      return null;
    }
    return tasks.value.find((entry) => entry.task.id === activeTaskId.value) || tasks.value[0];
  });

  const activeTaskSummary = computed<TaskSummary | null>(() => activeTask.value?.summary || null);
  const activeTaskEvidence = computed<EvidenceItem[]>(() => {
    if (activeTaskSummary.value?.evidence_items.length) {
      return activeTaskSummary.value.evidence_items;
    }
    return activeTask.value?.evidenceItems || [];
  });
  const activeTaskPapers = computed<PaperRecord[]>(() => {
    if (activeTaskSummary.value?.paper_records.length) {
      return activeTaskSummary.value.paper_records;
    }
    return activeTask.value?.papers || [];
  });
  const reportMarkdown = computed(() => finalReport.value?.markdown || "");
  const hasPendingRequest = computed(() => pendingRequest.value !== null);

  function normalizeRequest(payload: ResearchRequest): ResearchRequest {
    return {
      topic: payload.topic.trim(),
      top_k_online: payload.top_k_online ?? DEFAULT_REQUEST.top_k_online,
      top_k_local: payload.top_k_local ?? DEFAULT_REQUEST.top_k_local,
      search_provider: payload.search_provider?.trim() ? payload.search_provider.trim() : null,
      notes: payload.notes?.trim() ? payload.notes.trim() : null
    };
  }

  function resetRuntime() {
    run.value = null;
    error.value = "";
    status.value = "尚未开始";
    logs.value = [];
    tasks.value = [];
    activeTaskId.value = "";
    finalReport.value = null;
    runtimeState.value = null;
  }

  function appendLog(message: string) {
    logs.value = [...logs.value, message];
  }

  function ensureRunFromEvent(event: ResearchStreamEvent) {
    if (event.run && typeof event.run === "object") {
      run.value = event.run as ResearchRun;
      topic.value = run.value.topic;
      return;
    }

    if (run.value || !event.run_id || !topic.value) {
      return;
    }

    const now = new Date().toISOString();
    run.value = {
      id: String(event.run_id),
      topic: topic.value,
      status: "created",
      created_at: now,
      updated_at: now
    };
  }

  function updateRunStatus(nextStatus: string | undefined) {
    if (!run.value || !nextStatus) {
      return;
    }
    run.value = {
      ...run.value,
      status: nextStatus as ResearchRun["status"],
      updated_at: new Date().toISOString()
    };
  }

  function ensureActiveTask() {
    if (!tasks.value.length) {
      activeTaskId.value = "";
      return;
    }
    if (!tasks.value.some((entry) => entry.task.id === activeTaskId.value)) {
      activeTaskId.value = tasks.value[0].task.id;
    }
  }

  function setActiveTask(taskId: string) {
    activeTaskId.value = taskId;
  }

  function queueResearch(payload: ResearchRequest) {
    const request = normalizeRequest(payload);
    lastRequest.value = { ...request };
    pendingRequest.value = { ...request };
    topic.value = request.topic;
  }

  function buildFallbackSummary(task: TodoTask, summaryMarkdown: string): TaskSummary {
    return {
      task_id: task.id,
      title: task.title,
      intent: task.intent,
      summary: summaryMarkdown,
      summary_markdown: summaryMarkdown,
      evidence_items: [],
      paper_records: []
    };
  }

  function buildFallbackReport(event: ResearchStreamEvent): ResearchReport {
    const markdown = String(event.markdown || "");
    return {
      id: String(event.report_id || ""),
      report_id: String(event.report_id || ""),
      topic: topic.value,
      markdown,
      task_summaries: Object.values(taskSummaryMap.value),
      citations: [],
      citation_items: [],
      created_at: new Date().toISOString()
    };
  }

  function syncReportSummaries(report: ResearchReport) {
    if (!report.task_summaries.length) {
      return;
    }

    tasks.value = tasks.value.map((entry) => {
      const summary = report.task_summaries.find((item) => item.task_id === entry.task.id);
      if (!summary) {
        return entry;
      }
      return {
        ...entry,
        summary,
        evidenceItems: summary.evidence_items,
        papers: summary.paper_records,
        task: {
          ...entry.task,
          summary: summary.summary,
          summary_markdown: summary.summary_markdown
        }
      };
    });
  }

  function applyEvent(event: ResearchStreamEvent) {
    switch (event.type) {
      case "run_created":
        run.value = event.run as ResearchRun;
        topic.value = run.value.topic;
        appendLog(`已创建研究运行：${run.value.topic}`);
        break;
      case "status":
        ensureRunFromEvent(event);
        updateRunStatus(typeof event.status === "string" ? event.status : undefined);
        status.value = String(event.message || event.status || "处理中");
        appendLog(status.value);
        break;
      case "research_resumed":
        ensureRunFromEvent(event);
        isRunning.value = true;
        if (event.runtime_state && typeof event.runtime_state === "object") {
          runtimeState.value = event.runtime_state as ResearchRuntimeState;
        }
        appendLog("已从 checkpoint 恢复研究运行");
        break;
      case "todo_list":
        tasks.value = ((event.tasks as TodoTask[]) || []).map((task) => ({
          task,
          papers: [],
          evidenceItems: [],
          summary: null
        }));
        ensureActiveTask();
        appendLog(`已规划 ${tasks.value.length} 个研究任务`);
        break;
      case "task_status": {
        const taskId = String(event.task_id);
        tasks.value = tasks.value.map((entry) =>
          entry.task.id === taskId
            ? {
                ...entry,
                task: {
                  ...entry.task,
                  status: String(event.status) as TodoTask["status"]
                }
              }
            : entry
        );
        ensureActiveTask();
        appendLog(String(event.message || `${taskId} 状态更新`));
        break;
      }
      case "task_summary": {
        const taskId = String(event.task_id);
        tasks.value = tasks.value.map((entry) => {
          if (entry.task.id !== taskId) {
            return entry;
          }
          const summaryMarkdown = String(event.summary_markdown || "");
          const summary =
            (event.summary as TaskSummary | undefined) ||
            buildFallbackSummary(entry.task, summaryMarkdown);
          return {
            ...entry,
            summary,
            evidenceItems: summary.evidence_items,
            papers: summary.paper_records,
            task: {
              ...entry.task,
              summary: summary.summary,
              summary_markdown: summary.summary_markdown
            }
          };
        });
        if (!activeTaskId.value) {
          activeTaskId.value = taskId;
        }
        appendLog(`任务总结已返回：${taskSummaryMap.value[taskId]?.title || taskId}`);
        break;
      }
      case "agent_step_started":
        appendLog(
          `主 Agent 步骤开始：${String(event.action || "unknown")}${
            event.title ? ` / ${String(event.title)}` : ""
          }`
        );
        break;
      case "agent_step_completed":
        appendLog(
          `主 Agent 步骤完成：${String(event.action || "unknown")} - ${String(
            event.summary || "完成"
          )}`
        );
        break;
      case "agent_step_failed":
        appendLog(
          `主 Agent 步骤失败：${String(event.action || "unknown")} - ${String(
            event.error || "失败"
          )}`
        );
        break;
      case "checkpoint_saved":
        appendLog(
          `已保存 checkpoint：${String(event.current_phase || "unknown")} / step ${String(
            event.step_count || 0
          )}`
        );
        break;
      case "task_result": {
        const taskId = String(event.task_id);
        tasks.value = tasks.value.map((entry) =>
          entry.task.id === taskId
            ? {
                task: event.task as TodoTask,
                papers: (event.papers as PaperRecord[]) || [],
                evidenceItems: (event.evidence_items as EvidenceItem[]) || [],
                summary: (event.summary as TaskSummary) || null
              }
            : entry
        );
        ensureActiveTask();
        appendLog(`任务完成：${taskSummaryMap.value[taskId]?.title || taskId}`);
        break;
      }
      case "report": {
        const report =
          (event.report as ResearchReport | undefined) || buildFallbackReport(event);
        finalReport.value = report;
        syncReportSummaries(report);
        appendLog("最终综述已生成");
        break;
      }
      case "final_report":
        finalReport.value = event.report as ResearchReport;
        syncReportSummaries(finalReport.value);
        appendLog("最终综述已生成");
        break;
      case "done":
        isRunning.value = false;
        updateRunStatus("completed");
        status.value = "研究完成";
        appendLog("研究流程结束");
        break;
      case "error":
        isRunning.value = false;
        updateRunStatus("failed");
        error.value = String(event.detail || "研究失败");
        appendLog(error.value);
        break;
      default:
        appendLog(`收到事件：${event.type}`);
        break;
    }
  }

  async function startResearch(payload: ResearchRequest) {
    const request = normalizeRequest(payload);
    lastRequest.value = { ...request };
    pendingRequest.value = null;
    topic.value = request.topic;
    resetRuntime();
    isRunning.value = true;
    status.value = "正在初始化研究流程";
    try {
      await runResearchStream(request, applyEvent);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "研究请求失败";
      appendLog(error.value);
      isRunning.value = false;
    }
  }

  async function startPendingResearch() {
    if (!pendingRequest.value) {
      return;
    }
    const request = { ...pendingRequest.value };
    pendingRequest.value = null;
    await startResearch(request);
  }

  async function resumeCurrentRun() {
    if (!run.value) {
      return;
    }
    error.value = "";
    isRunning.value = true;
    status.value = "正在恢复研究流程";
    try {
      await resumeResearchStream(run.value.id, applyEvent);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "恢复研究请求失败";
      appendLog(error.value);
      isRunning.value = false;
    }
  }

  return {
    run,
    topic,
    lastRequest,
    pendingRequest,
    hasPendingRequest,
    isRunning,
    error,
    status,
    logs,
    tasks,
    activeTaskId,
    activeTask,
    activeTaskSummary,
    activeTaskEvidence,
    activeTaskPapers,
    finalReport,
    runtimeState,
    reportMarkdown,
    completedCount,
    canResumeCurrentRun,
    taskSummaryMap,
    resetRuntime,
    queueResearch,
    setActiveTask,
    startResearch,
    startPendingResearch,
    resumeCurrentRun
  };
});
