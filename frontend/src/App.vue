<template>
  <div class="app-shell" :class="{ 'app-shell-collapsed': sidebarCollapsed }">
    <div class="app-body">
      <aside class="sidebar" aria-label="主导航">
        <RouterLink class="brand-lockup" to="/knowledge" aria-label="返回知识库">
          <span class="brand-mark" aria-hidden="true">
            <Sparkles :size="18" stroke-width="2.4" />
          </span>
          <span class="brand-copy">
            <strong>PaperDesk</strong>
            <small>智能论文助手</small>
          </span>
        </RouterLink>

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
        <ResearchView v-show="isResearchRoute" />
        <RouterView v-slot="{ Component, route }">
          <component v-if="route.name !== 'research'" :is="Component" />
        </RouterView>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { useRoute } from "vue-router";
import {
  FileText,
  Library,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  Workflow
} from "lucide-vue-next";

import ResearchView from "./views/ResearchView.vue";

const route = useRoute();
const sidebarCollapsed = ref(window.localStorage.getItem("paperdesk-sidebar-collapsed") === "true");
const isResearchRoute = computed(() => route.name === "research");

watch(sidebarCollapsed, (value) => {
  window.localStorage.setItem("paperdesk-sidebar-collapsed", String(value));
});

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value;
}
</script>
