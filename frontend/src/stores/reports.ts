import { defineStore } from "pinia";
import { ref } from "vue";

import { getReport, listReports } from "../services/api";
import type { ReportListItem, ResearchReport } from "../types/models";

export const useReportStore = defineStore("reports", () => {
  const reports = ref<ReportListItem[]>([]);
  const activeReport = ref<ResearchReport | null>(null);
  const loading = ref(false);
  const error = ref("");

  async function refreshReports() {
    loading.value = true;
    error.value = "";
    try {
      reports.value = await listReports();
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载报告列表失败";
    } finally {
      loading.value = false;
    }
  }

  async function loadReport(reportId: string) {
    loading.value = true;
    error.value = "";
    try {
      activeReport.value = await getReport(reportId);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "加载报告失败";
    } finally {
      loading.value = false;
    }
  }

  return {
    reports,
    activeReport,
    loading,
    error,
    refreshReports,
    loadReport
  };
});

