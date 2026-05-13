import { createApp } from "vue";
import { createPinia } from "pinia";
import { createRouter, createWebHistory } from "vue-router";

import App from "./App.vue";
import KnowledgeView from "./views/KnowledgeView.vue";
import LibraryView from "./views/LibraryView.vue";
import ResearchView from "./views/ResearchView.vue";
import ReportView from "./views/ReportView.vue";
import "./style.css";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/knowledge" },
    { path: "/library", name: "library", component: LibraryView },
    { path: "/knowledge", name: "knowledge", component: KnowledgeView },
    { path: "/research", name: "research", component: ResearchView },
    { path: "/reports", name: "reports", component: ReportView }
  ]
});

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
