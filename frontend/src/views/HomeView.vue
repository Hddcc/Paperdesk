<template>
  <section class="page-shell home-workbench">
    <ResearchLaunchPanel
      title="启动一轮新的研究"
      description="先明确主题、检索范围和补充说明，再进入研究工作台持续查看进展。"
      submit-label="进入研究工作台"
      :initial-request="store.lastRequest"
      @submit="handleSubmit"
    />

    <article class="hero-card hero-side">
      <p class="eyebrow">PaperDesk</p>
      <h1>把研究主题拆成清晰、可跟踪的工作台</h1>
      <p class="lead">
        从本地论文库、在线检索到阶段性总结与最终综述，整个研究过程都会保留在同一套界面里。
      </p>

      <div class="info-stack">
        <section class="info-card">
          <h2>工作流会发生什么</h2>
          <ul class="bullet-list">
            <li>自动拆解 TODO 任务，并持续更新状态</li>
            <li>合并在线论文与本地文档证据</li>
            <li>逐步产出任务总结，最后形成完整综述</li>
          </ul>
        </section>

        <section class="info-card">
          <h2>你也可以先做这些准备</h2>
          <div class="hero-actions">
            <RouterLink class="button-secondary" to="/library">整理本地论文库</RouterLink>
            <RouterLink class="button-secondary" to="/knowledge">进入知识页</RouterLink>
            <RouterLink class="button-secondary" to="/reports">回看历史报告</RouterLink>
          </div>
        </section>
      </div>
    </article>
  </section>
</template>

<script setup lang="ts">
import { useRouter } from "vue-router";

import ResearchLaunchPanel from "../components/ResearchLaunchPanel.vue";
import { useResearchStore } from "../stores/research";
import type { ResearchRequest } from "../types/models";

const router = useRouter();
const store = useResearchStore();

async function handleSubmit(payload: ResearchRequest) {
  store.queueResearch(payload);
  await router.push("/research");
}
</script>
