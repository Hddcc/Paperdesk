export type ResearchRunStatus =
  | "created"
  | "planning"
  | "searching_online"
  | "retrieving_local"
  | "summarizing_task"
  | "writing_report"
  | "completed"
  | "failed"
  | "cancelled";

export type EvidenceSourceType = "online_paper" | "local_document";

export interface ResearchRequest {
  topic: string;
  top_k_online?: number;
  top_k_local?: number;
  notes?: string | null;
}

export interface TodoTask {
  id: string;
  title: string;
  intent: string;
  query: string;
  status: ResearchRunStatus;
  summary?: string | null;
}

export interface PaperRecord {
  title: string;
  authors: string[];
  year?: number | null;
  abstract: string;
  url?: string | null;
  doi?: string | null;
  source_type: EvidenceSourceType;
}

export interface LibraryDocument {
  id: string;
  filename: string;
  display_name: string;
  file_path: string;
  status: string;
  uploaded_at: string;
}

export interface EvidenceItem {
  id: string;
  source_type: EvidenceSourceType;
  source_id: string;
  quote: string;
  citation_label: string;
  metadata: Record<string, unknown>;
}

export interface TaskSummary {
  task_id: string;
  title: string;
  intent: string;
  summary: string;
  evidence_items: EvidenceItem[];
  paper_records: PaperRecord[];
}

export interface ResearchReport {
  id: string;
  topic: string;
  markdown: string;
  task_summaries: TaskSummary[];
  citations: string[];
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

export interface ResearchStreamEvent {
  type: string;
  [key: string]: unknown;
}

