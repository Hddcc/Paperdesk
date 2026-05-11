import { defineStore } from "pinia";
import { ref } from "vue";

import { analyzePapers, askKnowledgeQuestion, curatePapers } from "../services/api";
import type {
  PaperAnalysisRequest,
  PaperAnalysisResponse,
  PaperCurationRequest,
  PaperCurationResponse,
  RagAnswer,
  RagAskRequest
} from "../types/models";

export const useKnowledgeStore = defineStore("knowledge", () => {
  const asking = ref(false);
  const analyzing = ref(false);
  const curating = ref(false);
  const error = ref("");
  const ragAnswer = ref<RagAnswer | null>(null);
  const analysisResult = ref<PaperAnalysisResponse | null>(null);
  const curationResult = ref<PaperCurationResponse | null>(null);

  async function ask(payload: RagAskRequest) {
    asking.value = true;
    error.value = "";
    try {
      ragAnswer.value = await askKnowledgeQuestion(payload);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "知识问答失败";
      throw err;
    } finally {
      asking.value = false;
    }
  }

  async function analyze(payload: PaperAnalysisRequest) {
    analyzing.value = true;
    error.value = "";
    try {
      analysisResult.value = await analyzePapers(payload);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "论文分析失败";
      throw err;
    } finally {
      analyzing.value = false;
    }
  }

  async function curate(payload: PaperCurationRequest) {
    curating.value = true;
    error.value = "";
    try {
      curationResult.value = await curatePapers(payload);
    } catch (err) {
      error.value = err instanceof Error ? err.message : "候选筛选失败";
      throw err;
    } finally {
      curating.value = false;
    }
  }

  return {
    asking,
    analyzing,
    curating,
    error,
    ragAnswer,
    analysisResult,
    curationResult,
    ask,
    analyze,
    curate
  };
});
