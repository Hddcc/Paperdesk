<template>
  <section class="hero-grid">
    <article class="hero-card hero-main">
      <p class="eyebrow">PaperDesk Skeleton</p>
      <h1>智能论文助手 00/01 可运行骨架</h1>
      <p class="lead">
        当前版本只实现固定工作流、前后端分层、SQLite 持久化、本地 PDF 管理、
        任务流展示和 Markdown 报告预览。在线论文检索与本地语义召回仍为 mock/stub。
      </p>
      <div class="hero-actions">
        <RouterLink class="button-primary" to="/research">进入研究工作台</RouterLink>
        <RouterLink class="button-secondary" to="/library">管理本地论文库</RouterLink>
      </div>
    </article>

    <article class="hero-card hero-side">
      <h2>本轮实现范围</h2>
      <ul class="bullet-list">
        <li>5 个 Agent 固定职责边界</li>
        <li>顺序执行的研究工作流</li>
        <li>文档上传 / 删除 / 列表</li>
        <li>SSE 实时研究过程展示</li>
        <li>报告落盘与历史报告查看</li>
      </ul>
    </article>
  </section>
</template>

