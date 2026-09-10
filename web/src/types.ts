export type Kind = "drug" | "target";
export type Model = "biomaster" | "drugclip" | "dtiam" | "conplex" | "frozen";
export type EntityRef = {
  id: string;
  kind: Kind;
  name: string;
  subtitle?: string;
  scored?: boolean;
  annotations?: Record<string, number>;
  classification?: string;
};
export type Evidence = Record<string, any>;
export type Entity = EntityRef & {
  identifiers: Evidence;
  description: string;
  properties: Evidence;
  known_targets: Evidence[];
  known_diseases: Evidence[];
  txgnn_diseases: Evidence[];
  pathways: Evidence[];
  target_diseases: Evidence[];
  pockets: Evidence[];
  experiments?: Evidence[];
  evidence_totals?: Record<string, number>;
  models: Evidence[];
  sources: Evidence[];
  missing: string[];
  structure_url?: string;
  molecule_url?: string;
  [key: string]: any;
};
export type Ranking = EntityRef & {
  rank: number | null;
  score: number | null;
  scores: Record<Model, number | null>;
  ranks: Record<Model, number | null>;
  denominators: Record<Model, number>;
  known_relation?: boolean;
  frozen_score?: number;
  frozen_rank?: number;
  frozen_denominator?: number;
  action?: string;
  mechanism?: string;
  pocket_source?: string;
  drugclip_scope?: string;
};
export type Rankings = {
  kind: Kind;
  id: string;
  model: Model;
  page: number;
  page_size: number;
  total: number;
  denominator: number;
  scope: string;
  auxiliary: boolean;
  items: Ranking[];
  source: string;
};
export type Summary = {
  project: string;
  version: string;
  counts: Record<string, number>;
  models: Evidence[];
  sources: Evidence[];
  featured: { drugs: EntityRef[]; targets: EntityRef[] };
  boundaries: string[];
  warnings?: string[];
};
export const MODEL_NAMES: Record<Model, string> = {
  biomaster: "ReTargetMap",
  drugclip: "DrugCLIP",
  dtiam: "DTIAM",
  conplex: "ConPLex",
  frozen: "冻结生产版",
};
export const MODEL_COLORS: Record<Model, string> = {
  biomaster: "#006C5A",
  drugclip: "#2459E8",
  dtiam: "#7136CF",
  conplex: "#A34D0C",
  frozen: "#526176",
};
export const MODELS: Model[] = ["biomaster", "drugclip", "dtiam", "conplex"];
export function fmt(value: unknown, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value)
    ? Number.isInteger(value)
      ? value.toLocaleString("en-US")
      : value.toFixed(digits)
    : value == null || value === ""
      ? "—"
      : String(value);
}
export function label(row: Evidence): string {
  return (
    row.name ||
    row.disease_name ||
    row.pathway_name ||
    row.label ||
    row.target_name ||
    row.gene_symbol ||
    row.id ||
    "未命名记录"
  );
}
export { api } from "./request";
