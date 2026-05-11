<template>
  <section class="task-list">
    <header class="section-head">
      <h3>任务列表</h3>
      <p>{{ completedCount }} / {{ tasks.length }} 已完成</p>
    </header>

    <div class="panel-body panel-scroll">
      <ul v-if="tasks.length">
        <li v-for="entry in tasks" :key="entry.task.id">
          <button
            class="task-card task-card-button"
            :class="{ 'task-card-active': entry.task.id === activeTaskId }"
            type="button"
            @click="$emit('select', entry.task.id)"
          >
            <div class="task-title-row">
              <strong class="card-title">{{ entry.task.title }}</strong>
              <span class="status-badge" :data-status="entry.task.status">
                {{ formatTaskStatus(entry.task.status) }}
              </span>
            </div>
            <p class="task-intent">{{ entry.task.intent }}</p>
          </button>
        </li>
      </ul>

      <p v-else class="empty-state">研究开始后，这里会生成拆解任务。</p>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { ResearchTaskState } from "../types/models";

defineProps<{
  tasks: ResearchTaskState[];
  completedCount: number;
  activeTaskId: string;
}>();

defineEmits<{
  select: [taskId: string];
}>();

function formatTaskStatus(value: string) {
  switch (value) {
    case "pending":
      return "待处理";
    case "in_progress":
      return "进行中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    default:
      return value;
  }
}
</script>
