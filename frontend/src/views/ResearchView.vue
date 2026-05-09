<template>
  <section class="research-layout">
    <article class="panel">
      <header class="section-head">
        <h2>研究工作台</h2>
        <p>固定工作流：主题拆解 → 在线检索 → 本地检索 → 任务总结 → 报告生成</p>
      </header>

      <form class="stack-form" @submit.prevent="submit">
        <label>
          <span>研究主题</span>
          <textarea v-model="topic" rows="4" placeholder="例如：RAG 系统中的评估方法"></textarea>
        </label>
        <button class="button-primary" :disabled="store.isRunning || !topic.trim()" type="submit">
          {{ store.isRunning ? "研究进行中..." : "开始研究" }}
        </button>
      </form>

      <div class="status-strip">
        <strong>当前状态：</strong>
        <span>{{ store.status }}</span>
      </div>
      <p v-if="store.error" class="error-text">{{ store.error }}</p>
    </article>

    <ResearchTaskList :tasks="store.tasks" :completed-count="store.completedCount" />

    <article class="panel">
      <header class="section-head">
        <h2>研究过程</h2>
        <p>这里展示 SSE 事件日志，方便回看固定工作流。</p>
      </header>
      <ol class="log-list">
        <li v-for="(log, index) in store.logs" :key="`${index}-${log}`">{{ log }}</li>
      </ol>
    </article>

    <article class="panel">
      <header class="section-head">
        <h2>最终综述</h2>
        <p>生成完成后可在“报告预览”页查看历史记录。</p>
      </header>
      <MarkdownPreview v-if="store.finalReport" :markdown="store.finalReport.markdown" />
      <p v-else class="empty-state">报告尚未生成。</p>
    </article>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import ResearchTaskList from "../components/ResearchTaskList.vue";
import { useResearchStore } from "../stores/research";

const store = useResearchStore();
const topic = ref("");

async function submit() {
  await store.startResearch({ topic: topic.value.trim() });
}
</script>

