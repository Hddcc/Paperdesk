<template>
  <div class="app-shell" :class="{ 'app-shell-collapsed': sidebarCollapsed }">
    <header class="app-topbar">
      <RouterLink class="brand-lockup" to="/knowledge" aria-label="返回知识库">
        <span class="brand-mark" aria-hidden="true">
          <Sparkles :size="18" stroke-width="2.4" />
        </span>
        <span class="brand-copy">
          <strong>PaperDesk</strong>
          <small>智能论文助手</small>
        </span>
      </RouterLink>

    </header>

    <div class="app-body">
      <aside class="sidebar" aria-label="主导航">
        <nav class="nav-list">
          <RouterLink to="/knowledge" class="nav-item">
            <span class="nav-icon" aria-hidden="true">
              <MessageSquareText :size="19" />
            </span>
            <span class="nav-label">知识库</span>
          </RouterLink>
          <RouterLink to="/library" class="nav-item">
            <span class="nav-icon" aria-hidden="true">
              <Library :size="19" />
            </span>
            <span class="nav-label">本地论文</span>
          </RouterLink>
          <RouterLink to="/research" class="nav-item">
            <span class="nav-icon" aria-hidden="true">
              <Workflow :size="19" />
            </span>
            <span class="nav-label">工作台</span>
          </RouterLink>
          <RouterLink to="/reports" class="nav-item">
            <span class="nav-icon" aria-hidden="true">
              <FileText :size="19" />
            </span>
            <span class="nav-label">报告</span>
          </RouterLink>
        </nav>
        <button
          class="sidebar-toggle"
          type="button"
          :aria-label="sidebarCollapsed ? '展开侧栏' : '收起侧栏'"
          @click="toggleSidebar"
        >
          <PanelLeftOpen v-if="sidebarCollapsed" :size="19" />
          <PanelLeftClose v-else :size="19" />
          <span class="nav-label">{{ sidebarCollapsed ? "展开" : "收起" }}</span>
        </button>
      </aside>

      <main class="main-content">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from "vue";
import {
  FileText,
  Library,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  Workflow
} from "lucide-vue-next";

const sidebarCollapsed = ref(false);

onMounted(() => {
  sidebarCollapsed.value = window.localStorage.getItem("paperdesk-sidebar-collapsed") === "true";
});

watch(sidebarCollapsed, (value) => {
  window.localStorage.setItem("paperdesk-sidebar-collapsed", String(value));
});

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value;
}
</script>
