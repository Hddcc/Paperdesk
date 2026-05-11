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
  indexed_at?: string | null;
  version?: number;
  created_at?: string;
  uploaded_at: string;
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
}

export interface ResearchRunDetail {
  run: ResearchRun;
  tasks: TodoTask[];
  task_summaries: TaskSummary[];
  report: ResearchReport | null;
}

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
