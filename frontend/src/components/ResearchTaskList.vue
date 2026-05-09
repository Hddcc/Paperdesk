<template>
  <section class="task-list">
    <header class="section-head">
      <h3>任务列表</h3>
      <p>{{ completedCount }} / {{ tasks.length }} 已完成</p>
    </header>
    <ul>
      <li v-for="entry in tasks" :key="entry.task.id" class="task-card">
        <div class="task-title-row">
          <strong>{{ entry.task.title }}</strong>
          <span class="status-badge">{{ entry.task.status }}</span>
        </div>
        <p>{{ entry.task.intent }}</p>
        <pre v-if="entry.summary" class="task-summary">{{ entry.summary.summary }}</pre>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
import type { TaskSummary, TodoTask } from "../types/models";

defineProps<{
  tasks: Array<{
    task: TodoTask;
    summary: TaskSummary | null;
  }>;
  completedCount: number;
}>();
</script>

