<template>
  <article class="markdown-preview" v-html="rendered"></article>
</template>

<script setup lang="ts">
import MarkdownIt from "markdown-it";
import { computed } from "vue";

const props = defineProps<{
  markdown: string;
}>();

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: false
});

const rendered = computed(() => md.render(props.markdown || ""));
</script>
