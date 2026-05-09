import { defineStore } from "pinia";
import { computed, ref } from "vue";

import { runResearchStream } from "../services/api";
import type {
  EvidenceItem,
  PaperRecord,
  ResearchReport,
  ResearchRequest,
  ResearchRun,
  ResearchStreamEvent,
  TaskSummary,
  TodoTask
} from "../types/models";

interface TaskRuntimeState {
  task: TodoTask;
  papers: PaperRecord[];
  evidenceItems: EvidenceItem[];
  summary: TaskSummary | null;
}

export const useResearchStore = defineStore("research", () => {
  const run = ref<ResearchRun | null>(null);
  const topic = ref("");
  const isRunning = ref(false);
  const error = ref("");
  const status = ref("尚未开始");
  const logs = ref<string[]>([]);
  const tasks = ref<TaskRuntimeState[]>([]);
  const finalReport = ref<ResearchReport | null>(null);

  const completedCount = computed(
    () => tasks.value.filter((entry) => entry.task.status === "completed").length
  );

  function reset() {
    run.value = null;
    error.value = "";
    status.value = "尚未开始";
    logs.value = [];
    tasks.value = [];
    finalReport.value = null;
  }

  function appendLog(message: string) {
    logs.value = [...logs.value, message];
  }

  function applyEvent(event: ResearchStreamEvent) {
    switch (event.type) {
      case "run_created":
        run.value = event.run as ResearchRun;
        appendLog(`已创建研究运行：${run.value.topic}`);
        break;
      case "status":
        status.value = String(event.message || event.status || "处理中");
        appendLog(status.value);
        break;
      case "todo_list":
        tasks.value = ((event.tasks as TodoTask[]) || []).map((task) => ({
          task,
          papers: [],
          evidenceItems: [],
          summary: null
        }));
        appendLog(`已规划 ${tasks.value.length} 个研究任务`);
        break;
      case "task_status": {
        const taskId = String(event.task_id);
        const entry = tasks.value.find((item) => item.task.id === taskId);
        if (entry) {
          entry.task.status = String(event.status) as TodoTask["status"];
        }
        appendLog(String(event.message || `${taskId} 状态更新`));
        break;
      }
      case "task_result": {
        const taskId = String(event.task_id);
        const entry = tasks.value.find((item) => item.task.id === taskId);
        if (entry) {
          entry.task = event.task as TodoTask;
          entry.papers = (event.papers as PaperRecord[]) || [];
          entry.evidenceItems = (event.evidence_items as EvidenceItem[]) || [];
          entry.summary = (event.summary as TaskSummary) || null;
        }
        appendLog(`任务完成：${entry?.task.title || taskId}`);
        break;
      }
      case "final_report":
        finalReport.value = event.report as ResearchReport;
        appendLog("最终综述已生成");
        break;
      case "done":
        isRunning.value = false;
        status.value = "研究完成";
        appendLog("研究流程结束");
        break;
      case "error":
        isRunning.value = false;
        error.value = String(event.detail || "研究失败");
        appendLog(error.value);
        break;
      default:
        appendLog(`收到事件：${event.type}`);
        break;
    }
  }

  async function startResearch(payload: ResearchRequest) {
    reset();
    topic.value = payload.topic;
    isRunning.value = true;
    status.value = "正在初始化研究流程";
    try {
      await runResearchStream(payload, applyEvent);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "研究请求失败";
      appendLog(error.value);
      isRunning.value = false;
    }
  }

  return {
    run,
    topic,
    isRunning,
    error,
    status,
    logs,
    tasks,
    finalReport,
    completedCount,
    reset,
    startResearch
  };
});

