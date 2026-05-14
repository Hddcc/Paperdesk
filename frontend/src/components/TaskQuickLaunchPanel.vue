<template>
  <section v-if="documentCount > 0" class="quick-launch-panel" :class="{ 'quick-launch-compact': compact }">
    <div class="quick-launch-head">
      <div>
        <span class="request-label">{{ contextLabel }}</span>
        <strong>{{ title }}</strong>
        <p>{{ description }}</p>
      </div>
    </div>

    <div class="quick-action-grid">
      <article v-for="action in visibleActions" :key="action.id" class="quick-action-card">
        <div class="quick-action-copy">
          <strong>{{ action.label }}</strong>
          <p>{{ action.description }}</p>
        </div>
        <div class="quick-action-buttons">
          <button class="button-secondary" type="button" :disabled="disabled" @click="emit('fill', action)">
            <PencilLine :size="15" />
            填入输入框
          </button>
          <button class="button-primary" type="button" :disabled="disabled" @click="emit('submit', action)">
            <Sparkles :size="15" />
            生成结果
          </button>
        </div>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { PencilLine, Sparkles } from "lucide-vue-next";

export interface QuickLaunchAction {
  id: string;
  label: string;
  description: string;
  prompt: string;
  notes: string;
  minDocuments: number;
  maxDocuments?: number;
}

const props = withDefaults(
  defineProps<{
    documentCount: number;
    hasUploadedContext?: boolean;
    disabled?: boolean;
    compact?: boolean;
    title?: string;
  }>(),
  {
    hasUploadedContext: false,
    disabled: false,
    compact: false,
    title: "基于已选材料继续"
  }
);

const emit = defineEmits<{
  fill: [action: QuickLaunchAction];
  submit: [action: QuickLaunchAction];
}>();

const contextLabel = computed(() =>
  props.hasUploadedContext ? "上传材料" : "已选论文"
);

const description = computed(() => {
  const materialName = props.hasUploadedContext ? "上传材料" : "论文";
  return props.documentCount === 1
    ? `已选 1 篇${materialName}，可以直接提问，也可以快速发起总结或方法分析。`
    : `已选 ${props.documentCount} 篇${materialName}，可以继续提问，也可以快速发起综述、对比或研究 brief。`;
});

const actions: QuickLaunchAction[] = [
  {
    id: "single-summary",
    label: "总结这篇",
    description: "生成一份面向阅读理解的论文总结。",
    prompt: "请总结这篇论文，重点说明研究问题、核心方法、主要贡献、实验结论、局限性和适用场景。",
    notes: "轻入口：基于单篇已选材料发起论文总结。",
    minDocuments: 1,
    maxDocuments: 1
  },
  {
    id: "single-question",
    label: "针对论文提问",
    description: "先把问题放入输入框，便于继续修改。",
    prompt: "请基于这篇论文回答：",
    notes: "轻入口：基于单篇已选材料继续问答。",
    minDocuments: 1,
    maxDocuments: 1
  },
  {
    id: "single-method",
    label: "提取方法要点",
    description: "聚焦方法流程、假设、适用场景与局限。",
    prompt: "请提取并解释这篇论文的核心方法、关键假设、适用场景和局限性。",
    notes: "轻入口：基于单篇已选材料提取方法要点。",
    minDocuments: 1,
    maxDocuments: 1
  },
  {
    id: "multi-review",
    label: "生成综述",
    description: "围绕已选文献组织主题脉络与代表工作。",
    prompt: "请基于已选文献生成一份综述，梳理研究主题、子方向、代表论文、核心方法脉络、趋势与不足。",
    notes: "轻入口：基于多篇已选材料生成综述。",
    minDocuments: 2
  },
  {
    id: "multi-comparison",
    label: "对比分析",
    description: "按研究问题、方法、证据与适用场景做对比。",
    prompt: "请对比分析已选文献，重点比较研究问题、方法路线、实验证据、结论差异和适用建议。",
    notes: "轻入口：基于多篇已选材料发起对比分析。",
    minDocuments: 2
  },
  {
    id: "multi-differences",
    label: "共同点与差异",
    description: "先找共识，再标出关键分歧和互补关系。",
    prompt: "请分析已选文献的共同点与差异，说明它们在问题设定、方法设计、实验结果和局限性上的共性与区别。",
    notes: "轻入口：基于多篇已选材料分析共同点与差异。",
    minDocuments: 2
  },
  {
    id: "multi-brief",
    label: "研究 brief",
    description: "生成下一步研究方向、路线和验证建议。",
    prompt: "请基于已选文献生成研究 brief，概括方向现状、关键问题、可行研究路线、后续验证方式和风险点。",
    notes: "轻入口：基于多篇已选材料生成研究 brief。",
    minDocuments: 2
  }
];

const visibleActions = computed(() =>
  actions.filter((action) => {
    if (props.documentCount < action.minDocuments) {
      return false;
    }
    return action.maxDocuments === undefined || props.documentCount <= action.maxDocuments;
  })
);
</script>
