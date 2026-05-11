<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div>
        <p class="eyebrow">PaperDesk</p>
        <h1>智能论文助手</h1>
      </div>

      <nav class="nav-list">
        <RouterLink to="/">总览</RouterLink>
        <RouterLink to="/library">本地论文库</RouterLink>
        <RouterLink to="/knowledge">知识问答</RouterLink>
        <RouterLink to="/research">研究工作台</RouterLink>
        <RouterLink to="/reports">报告预览</RouterLink>
      </nav>
    </aside>

    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>
