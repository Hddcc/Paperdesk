export type ResearchRunStatus =
  | "created"
  | "planning"
  | "running_task"
  | "writing_report"
  | "completed"
  | "failed"
  | "cancelled";

export type TodoTaskStatus = "pending" | "in_progress" | "completed" | "failed";

export type EvidenceSourceType = "online_paper" | "local_document";

export interface ResearchRequest {
  topic: string;
  top_k_online?: number;
  top_k_local?: number;
  search_provider?: string | null;
  notes?: string | null;
  input_modes?: ResearchInputMode[];
  selected_document_ids?: string[];
}

export type ResearchInputMode = "prompt" | "uploaded_file" | "knowledge_base";
export type ResearchTaskType =
  | "qa"
  | "paper_summary"
  | "multi_paper_review"
  | "comparison"
  | "method_explainer"
  | "research_brief";
export type ResearchEvidencePolicy = "local_first" | "online_first" | "local_only" | "online_supplement";
export type ResearchExecutionRoute =
  | "knowledge_qa"
  | "single_paper_summary"
  | "main_agent_review"
  | "comparison_analysis"
  | "method_explanation"
  | "research_brief";
export type ResearchArtifactProtocolType =
  | "qa"
  | "paper_summary"
  | "review"
  | "comparison"
  | "method_explainer"
  | "research_brief";

export interface ResearchArtifactProtocol {
  protocol_type: ResearchArtifactProtocolType;
  title: string;
  required_sections: string[];
  citation_required: boolean;
}

export interface ResearchTaskRoute {
  task_type: ResearchTaskType;
  input_modes: ResearchInputMode[];
  evidence_policy: ResearchEvidencePolicy;
  execution_route: ResearchExecutionRoute;
  artifact_protocol: ResearchArtifactProtocol;
  selected_document_ids: string[];
  needs_local_knowledge: boolean;
  needs_online_search: boolean;
  use_main_agent_loop: boolean;
  allow_single_pass: boolean;
  rationale: string;
}

export interface ResearchTaskState {
  task: TodoTask;
  papers: PaperRecord[];
  evidenceItems: EvidenceItem[];
  summary: TaskSummary | null;
}

export interface TodoTask {
  id: string;
  title: string;
  intent: string;
  query: string;
  status: TodoTaskStatus;
  summary?: string | null;
  summary_markdown?: string | null;
}

export interface PaperRecord {
  paper_id?: string | null;
  title: string;
  authors: string[];
  abstract?: string | null;
  year?: number | null;
  venue?: string | null;
  url?: string | null;
  doi?: string | null;
  source?: string;
  source_type: EvidenceSourceType;
}

export interface LibraryDocument {
  id: string;
  filename: string;
  display_name: string;
  title?: string | null;
  file_path: string;
  sha256?: string;
  page_count?: number;
  status: string;
  parser_status?: string;
  failure_reason?: string | null;
  indexed_at?: string | null;
  version?: number;
  created_at?: string;
  uploaded_at: string;
  categories: DocumentCategory[];
}

export interface DocumentCategory {
  id: string;
  name: string;
  color?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentCategoryCreateRequest {
  name: string;
  color?: string | null;
}

export interface DocumentCategoryUpdateRequest {
  name?: string | null;
  color?: string | null;
}

export interface DocumentCategoryAssignmentRequest {
  category_ids: string[];
}

export interface EvidenceItem {
  id: string;
  evidence_id?: string;
  source_type: EvidenceSourceType;
  source_id: string;
  title?: string;
  snippet?: string;
  quote: string;
  citation_label: string;
  url?: string | null;
  document_id?: string | null;
  page_number?: number | null;
  score?: number | null;
  metadata: Record<string, unknown>;
}

export interface TaskSummary {
  task_id: string;
  title: string;
  intent: string;
  summary: string;
  summary_markdown?: string | null;
  evidence_items: EvidenceItem[];
  paper_records: PaperRecord[];
}

export interface CitationRecord {
  citation_label: string;
  source_type: string;
  title: string;
  url?: string | null;
  doi?: string | null;
  document_id?: string | null;
  page_number?: number | null;
}

export interface ResearchReport {
  id: string;
  report_id?: string | null;
  topic: string;
  markdown: string;
  task_summaries: TaskSummary[];
  citations: string[];
  citation_items?: CitationRecord[];
  created_at: string;
}

export interface ReportListItem {
  id: string;
  topic: string;
  created_at: string;
}

export interface ResearchRun {
  id: string;
  topic: string;
  status: ResearchRunStatus;
  created_at: string;
  updated_at: string;
  stop_reason?: string | null;
  last_checkpoint_at?: string | null;
}

export type ResearchActionType =
  | "plan"
  | "search_online"
  | "search_local"
  | "summarize_evidence"
  | "revise_plan"
  | "finalize_report"
  | "finish"
  | "fail";

export type PlannerProviderType = "rule_based" | "llm_candidate" | "hybrid_candidate";

export type ResearchPlanOperationType =
  | "rewrite_query"
  | "insert_item"
  | "split_item"
  | "merge_items"
  | "reorder_items"
  | "close_item";

export interface ResearchToolStrategy {
  strategy_id: string;
  action_type: ResearchActionType;
  label: string;
  parameters: Record<string, unknown>;
  rationale: string;
}

export interface ResearchPlanOperation {
  operation_type: ResearchPlanOperationType;
  target_task_id?: string | null;
  source_task_ids: string[];
  new_task_id?: string | null;
  title?: string | null;
  intent?: string | null;
  query?: string | null;
  priority?: number | null;
  ordered_task_ids: string[];
  reason: string;
  applied_at?: string | null;
}

export interface ResearchRuntimeStep {
  step_id: string;
  action: ResearchActionType;
  task_id?: string | null;
  attempt: number;
  selected_tool?: string | null;
  tool_strategy?: ResearchToolStrategy | null;
  reason: string;
  status: "running" | "completed" | "failed";
  started_at: string;
}

export type ResearchToolResultClassification =
  | "success_sufficient"
  | "success_insufficient"
  | "retryable_error"
  | "non_retryable_error"
  | "no_increment";

export interface ResearchToolCallRecord {
  step_id: string;
  action: ResearchActionType;
  task_id?: string | null;
  status: "completed" | "failed" | "skipped";
  summary: string;
  selected_tool?: string | null;
  tool_strategy?: ResearchToolStrategy | null;
  decision_reason: string;
  result_classification?: ResearchToolResultClassification | null;
  planner_provider?: PlannerProviderType | null;
  planner_fallback_used: boolean;
  plan_operations: ResearchPlanOperation[];
  retryable: boolean;
  error?: string | null;
  paper_count: number;
  evidence_count: number;
  created_at: string;
}

export interface ResearchEvidenceBufferItem {
  task_id: string;
  paper_records: PaperRecord[];
  evidence_items: EvidenceItem[];
  compacted_evidence: ResearchCompactedEvidenceItem[];
  evidence_assessment: ResearchEvidenceAssessment;
  online_completed: boolean;
  local_completed: boolean;
  degraded: boolean;
}

export type ResearchContextStage =
  | "normal"
  | "evidence_compacted"
  | "history_compacted"
  | "truncated";

export interface ResearchCompactedEvidenceItem {
  task_id: string;
  source_key: string;
  source_type: string;
  citation: string;
  title: string;
  page_number?: number | null;
  excerpt: string;
  relevance: string;
  coverage: string[];
  potential_conflict: boolean;
  visible: boolean;
}

export interface ResearchEvidenceAssessment {
  total_item_count: number;
  paper_count: number;
  local_evidence_count: number;
  relevant_item_count: number;
  visible_item_count: number;
  compacted_item_count: number;
  sufficiency_score: number;
  relevance_score: number;
  diversity_score: number;
  coverage: string[];
  conflict_detected: boolean;
  has_relevant_evidence: boolean;
  rationale: string;
}

export interface ResearchContextState {
  stage: ResearchContextStage;
  estimated_tokens: number;
  budget_tokens: number;
  sources: string[];
  last_compacted_at?: string | null;
  active_task_id?: string | null;
  visible_step_count: number;
  evidence_items_compacted: number;
  history_compacted: boolean;
}

export interface ResearchPlanItem {
  task_id: string;
  title: string;
  intent: string;
  query: string;
  objective: string;
  done_criteria: string;
  status: TodoTaskStatus;
  priority: number;
  suggested_tools: string[];
  required_evidence: string[];
  attempt_count: number;
  notes: string[];
  revise_count: number;
  query_history: string[];
  summary?: string | null;
  summary_markdown?: string | null;
  degraded: boolean;
}

export interface ResearchActionDecision {
  action_type: ResearchActionType;
  selected_tool?: string | null;
  tool_strategy?: ResearchToolStrategy | null;
  reason: string;
  target_task_id?: string | null;
}

export interface ResearchRuntimeState {
  run_id: string;
  goal: string;
  current_phase:
    | "planning"
    | "executing"
    | "summarizing"
    | "writing_report"
    | "completed"
    | "failed";
  task_route?: ResearchTaskRoute | null;
  plan_items: ResearchPlanItem[];
  completed_items: string[];
  active_step?: ResearchRuntimeStep | null;
  tool_history: ResearchToolCallRecord[];
  evidence_buffer: ResearchEvidenceBufferItem[];
  context_state: ResearchContextState;
  working_summary: string;
  failure_count: number;
  replan_count: number;
  no_progress_count: number;
  same_tool_streak: number;
  last_tool_signature?: string | null;
  last_decision?: ResearchActionDecision | null;
  plan_revision_history: ResearchPlanOperation[];
  last_plan_operation?: ResearchPlanOperation | null;
  planner_provider: PlannerProviderType;
  planner_fallback_used: boolean;
  stop_reason?: string | null;
  last_checkpoint_at?: string | null;
  step_count: number;
  report_id?: string | null;
}

export interface ResearchRunDetail {
  run: ResearchRun;
  tasks: TodoTask[];
  task_summaries: TaskSummary[];
  runtime_state?: ResearchRuntimeState | null;
  report: ResearchReport | null;
  task_route?: ResearchTaskRoute | null;
}

export type KnowledgeRetrievalStatus = "ready" | "skipped" | "degraded" | "unavailable";

export interface RagAskRequest {
  question: string;
  document_ids?: string[];
  top_k?: number;
  notes?: string | null;
}

export interface RagAnswer {
  answer: string;
  citations: string[];
  sources: string[];
  pages: number[];
  retrieval_count: number;
  confidence?: number | null;
  evidence_items: EvidenceItem[];
}

export type ChatAttachmentKind = "image" | "uploaded_pdf" | "library_document";
export type ChatMessageRole = "user" | "assistant" | "system";
export type MemoryRecordType = "user" | "feedback" | "project" | "reference";
export type ContextStage = "normal" | "evidence_compacted" | "history_compacted" | "truncated";

export interface ChatAttachment {
  id: string;
  kind: ChatAttachmentKind;
  display_name: string;
  mime_type?: string | null;
  document_id?: string | null;
  data_url?: string | null;
  file_path?: string | null;
  status?: string | null;
  metadata: Record<string, unknown>;
}

export interface MemoryHit {
  id: string;
  memory_type: MemoryRecordType;
  summary: string;
  detail?: string | null;
  source_kind?: string | null;
  source_id?: string | null;
  status: string;
  last_verified_at?: string | null;
}

export interface MemorySnapshot {
  items: MemoryHit[];
  refreshed_at: string;
}

export interface ChatContextState {
  stage: ContextStage;
  estimated_tokens: number;
  budget_tokens: number;
  sources: string[];
  last_compacted_at?: string | null;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: ChatMessageRole;
  content: string;
  status: string;
  retrieval_status?: KnowledgeRetrievalStatus | null;
  warning?: string | null;
  citations: string[];
  used_document_ids: string[];
  memory_hits: MemoryHit[];
  attachments: ChatAttachment[];
  saved_report_id?: string | null;
  agent_trace_id?: string | null;
  action_status?: string | null;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_preview?: string | null;
}

export interface ChatSessionDetail {
  session: ChatSession;
  messages: ChatMessage[];
  memory_snapshot: MemorySnapshot;
  context_state: ChatContextState;
}

export interface ChatSessionCreateRequest {
  title?: string | null;
}

export interface ChatMessageRequest {
  content: string;
  attachments?: ChatAttachment[];
  selected_document_ids?: string[];
}

export interface ChatSendResponse {
  session: ChatSession;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  memory_snapshot: MemorySnapshot;
  context_state: ChatContextState;
  library_mutated?: boolean;
}

export type ChatStreamEvent =
  | { type: "status"; status: string; message?: string }
  | { type: "assistant_delta"; delta: string }
  | { type: "done"; response: ChatSendResponse }
  | { type: "error"; message: string };

export interface PaperAnalysisRequest {
  document_ids: string[];
  mode: "single" | "compare";
  question?: string | null;
}

export interface PaperAnalysisSection {
  title: string;
  content: string;
}

export interface PaperAnalysisResponse {
  mode: "single" | "compare";
  answer: string;
  sections: PaperAnalysisSection[];
  citations: string[];
  evidence_items: EvidenceItem[];
  retrieval_count: number;
}

export interface PaperCurationRequest {
  topic: string;
  search_provider?: string | null;
  top_k_online?: number;
}

export interface PaperCurationItem {
  paper: PaperRecord;
  decision: "recommended" | "consider" | "skip";
  reason: string;
}

export interface PaperCurationResponse {
  items: PaperCurationItem[];
}

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}
