<template>
  <article class="launch-panel" :class="{ 'launch-panel-compact': compact }">
    <header class="section-head">
      <div>
        <h2>{{ title }}</h2>
        <p>{{ description }}</p>
      </div>
    </header>

    <form class="stack-form" @submit.prevent="emitSubmit">
      <label>
        <span>研究主题</span>
        <textarea
          v-model="topic"
          :rows="compact ? 3 : 5"
          placeholder="例如：RAG 系统中的评估方法"
        ></textarea>
      </label>

      <div class="request-grid">
        <label>
          <span>在线论文数量</span>
          <input v-model.number="topKOnline" max="10" min="1" type="number" />
        </label>
        <label>
          <span>本地证据数量</span>
          <input v-model.number="topKLocal" max="10" min="1" type="number" />
        </label>
        <label>
          <span>检索来源</span>
          <select v-model="searchProvider">
            <option value="">自动选择</option>
            <option value="all">全部来源</option>
            <option value="openalex">仅 OpenAlex</option>
            <option value="arxiv">仅 arXiv</option>
          </select>
        </label>
      </div>

      <label>
        <span>补充说明</span>
        <textarea
          v-model="notes"
          :rows="compact ? 2 : 3"
          placeholder="可选：限定研究角度、关注的数据集、要对比的方法等。"
        ></textarea>
      </label>

      <button class="button-primary" :disabled="disabled || !canSubmit" type="submit">
        <Rocket :size="16" />
        {{ disabled ? "研究进行中..." : submitLabel }}
      </button>
    </form>
  </article>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { Rocket } from "lucide-vue-next";

import type { ResearchRequest } from "../types/models";

const props = withDefaults(
  defineProps<{
    title: string;
    description: string;
    submitLabel: string;
    disabled?: boolean;
    compact?: boolean;
    initialRequest?: Partial<ResearchRequest>;
  }>(),
  {
    disabled: false,
    compact: false,
    initialRequest: () => ({})
  }
);

const emit = defineEmits<{
  submit: [payload: ResearchRequest];
}>();

const topic = ref("");
const topKOnline = ref(3);
const topKLocal = ref(3);
const searchProvider = ref("");
const notes = ref("");
const canSubmit = computed(() => topic.value.trim().length >= 3);

watch(
  () => props.initialRequest,
  (request) => {
    topic.value = request.topic || "";
    topKOnline.value = request.top_k_online ?? 3;
    topKLocal.value = request.top_k_local ?? 3;
    searchProvider.value = request.search_provider || "";
    notes.value = request.notes || "";
  },
  { immediate: true, deep: true }
);

function emitSubmit() {
  if (!canSubmit.value) {
    return;
  }

  emit("submit", {
    topic: topic.value.trim(),
    top_k_online: clampCount(topKOnline.value),
    top_k_local: clampCount(topKLocal.value),
    search_provider: searchProvider.value || null,
    notes: notes.value.trim() || null,
    input_modes: props.initialRequest.input_modes?.length
      ? [...props.initialRequest.input_modes]
      : ["prompt"],
    selected_document_ids: props.initialRequest.selected_document_ids
      ? [...props.initialRequest.selected_document_ids]
      : []
  });
}

function clampCount(value: number) {
  return Math.max(1, Math.min(10, value || 3));
}
</script>
