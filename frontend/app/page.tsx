"use client";

import React, { useState, useRef, useCallback } from "react";
import axios from "axios";
import {
  ShieldCheck,
  ShieldAlert,
  Upload,
  FileText,
  Activity,
  AlertTriangle,
  Database,
  Bug,
  Percent,
  Search,
  Download,
  Loader2,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type { ValueType } from "recharts/types/component/DefaultTooltipContent";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */
interface ImageContributor {
  feature_index: number;
  deviation_score: number;
}

interface TabularContributor {
  name: string;
  error: number;
}

type Explanation =
  | { type: "image"; top_contributors: ImageContributor[]; error?: string }
  | { type: "tabular"; top_contributors: TabularContributor[] };

interface FlaggedItem {
  id: string;
  flagged_by: string[];
  explanation?: Explanation;
}

interface ScanResults {
  total_samples: number;
  poisoned_samples: number;
  anomaly_percentage: number;
  model_breakdown: Record<string, number>;
  flagged_items: FlaggedItem[];
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1GB

const CHART_COLORS = {
  clean: "#22d3ee",
  poisoned: "#f43f5e",
};

/** Per-model badge + bar colors */
const MODEL_COLORS: Record<string, { bg: string; text: string; bar: string }> = {
  Autoencoder:        { bg: "bg-violet-500/15", text: "text-violet-400", bar: "#a78bfa" },
  "Deep SVDD":        { bg: "bg-amber-500/15",  text: "text-amber-400",  bar: "#fbbf24" },
  "Isolation Forest": { bg: "bg-sky-500/15",    text: "text-sky-400",    bar: "#38bdf8" },
};

const DEFAULT_MODEL_STYLE = { bg: "bg-rose-500/15", text: "text-rose-400", bar: "#f43f5e" };

/* ------------------------------------------------------------------ */
/* Page Component                                                      */
/* ------------------------------------------------------------------ */
export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedItem, setSelectedItem] = useState<FlaggedItem | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [results, setResults] = useState<ScanResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [filterQuery, setFilterQuery] = useState("");

  /* ---- File handling -------------------------------------------- */
  const handleFile = useCallback((f: File | null) => {
    if (f && f.size > MAX_FILE_SIZE) {
      setFile(null);
      setResults(null);
      setError("File too large. Maximum limit is 1GB.");
      return;
    }

    setFile(f);
    setResults(null);
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const dropped = e.dataTransfer.files?.[0] ?? null;
      handleFile(dropped);
    },
    [handleFile]
  );

  /* ---- Scan API call -------------------------------------------- */
  const scanDataset = async () => {
    if (!file) return;
    setIsScanning(true);
    setResults(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const { data } = await axios.post<ScanResults>(
        `${API_URL}/scan-dataset`,
        formData
      );
      setResults(data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail ??
            err.message ??
            "Scan failed. Is the backend running?"
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setIsScanning(false);
    }
  };

  /* ---- Download File -------------------------------------------- */
  const handleDownloadSanitized = async () => {
    if (!file || !results) return;
    setIsDownloading(true);
    try {
      const flaggedIds = results.flagged_items.map((i) => i.id);
      
      const formData = new FormData();
      formData.append("file", file);
      formData.append("flagged_items", JSON.stringify(flaggedIds));

      const response = await axios.post(`${API_URL}/download-sanitized`, formData, {
        responseType: "blob",
      });

      const blob = new Blob([response.data]);
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      const extension = file.name.split(".").pop() || "";
      link.setAttribute("download", `sanitized_${file.name.replace(`.${extension}`, "")}.${extension}`);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail ?? err.message ?? "Download failed.");
      } else {
        setError("An unexpected error occurred during download.");
      }
    } finally {
      setIsDownloading(false);
    }
  };

  /* ---- Derived data --------------------------------------------- */
  const chartData = results
    ? [
        {
          name: "Clean",
          value: results.total_samples - results.poisoned_samples,
        },
        { name: "Poisoned", value: results.poisoned_samples },
      ]
    : [];

  const cleanPct = results
    ? (100 - results.anomaly_percentage).toFixed(2)
    : null;

  const filteredItems = results
    ? results.flagged_items.filter((item) =>
        item.id.toLowerCase().includes(filterQuery.toLowerCase())
      )
    : [];

  const barChartData = results
    ? Object.entries(results.model_breakdown).map(([name, count]) => ({
        name,
        flagged: count,
        fill: (MODEL_COLORS[name] ?? DEFAULT_MODEL_STYLE).bar,
      }))
    : [];

  /* ---------------------------------------------------------------- */
  /* Render                                                           */
  /* ---------------------------------------------------------------- */
  return (
    <main className="flex-1 flex flex-col">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-6 py-5 flex items-center gap-4">
          <div className="rounded-xl bg-gradient-to-br from-sky-500 to-cyan-400 p-2.5 shadow-lg shadow-sky-500/20">
            <ShieldCheck className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              Distill
            </h1>
            <p className="text-sm text-slate-400 mt-0.5">
              Universal Data Sanitization &amp; Poisoning Detection
            </p>
          </div>
          {/* status pill */}
          <span className="ml-auto inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400 ring-1 ring-emerald-500/20">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            System Online
          </span>
        </div>
      </header>

      {/* ── Body ──────────────────────────────────────────────────── */}
      <div className="mx-auto w-full max-w-7xl px-6 py-10 flex flex-col gap-10">
        {/* ── Upload Zone ──────────────────────────────────────────── */}
        <section
          id="upload-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className="group relative cursor-pointer rounded-2xl border-2 border-dashed border-[var(--color-border)] bg-[var(--color-surface)] hover:border-sky-500/60 transition-all duration-300 p-10 text-center"
        >
          {/* subtle gradient shimmer on hover */}
          <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-br from-sky-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

          <input
            ref={fileInputRef}
            id="file-input"
            type="file"
            accept=".csv,.zip"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          />

          {file ? (
            <div className="flex flex-col items-center gap-3 animate-fade-up">
              <FileText className="h-12 w-12 text-sky-400" />
              <p className="text-lg font-semibold text-white">{file.name}</p>
              <p className="text-sm text-slate-400">
                {(file.size / 1024).toFixed(1)} KB — Click or drop to replace
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Upload className="h-12 w-12 text-slate-500 group-hover:text-sky-400 transition-colors" />
              <p className="text-lg font-medium text-slate-300">
                Drag &amp; drop your file here
              </p>
              <p className="text-sm text-slate-500">
                Supports <span className="text-sky-400">.csv</span> and{" "}
                <span className="text-sky-400">.zip</span> (image archives)
              </p>
            </div>
          )}
        </section>

        {/* ── Scan Button ─────────────────────────────────────────── */}
        <div className="flex justify-center">
          <button
            id="scan-button"
            disabled={!file || isScanning}
            onClick={scanDataset}
            className="relative inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-cyan-500 px-8 py-3.5 text-sm font-bold text-white shadow-lg shadow-sky-600/25 transition-all hover:shadow-sky-500/40 hover:scale-[1.03] active:scale-[0.98] disabled:opacity-40 disabled:pointer-events-none disabled:shadow-none"
          >
            <Activity className="h-4 w-4" />
            {isScanning ? "Scanning…" : "Scan Dataset"}
          </button>
        </div>

        {/* ── Error ──────────────────────────────────────────────── */}
        {error && (
          <div
            id="error-banner"
            className="rounded-xl border border-red-500/30 bg-red-500/10 px-5 py-4 text-sm text-red-300 flex items-start gap-3 animate-fade-up"
          >
            <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5 text-red-400" />
            <span>{error}</span>
          </div>
        )}

        {/* ── Scanning Animation ─────────────────────────────────── */}
        {isScanning && (
          <div className="flex flex-col items-center gap-6 py-16 animate-fade-up">
            {/* scanner box */}
            <div className="relative h-40 w-40 rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] overflow-hidden">
              {/* sweep line */}
              <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-sky-400 to-transparent animate-scan-line" />
              {/* center icon */}
              <div className="flex h-full items-center justify-center">
                <ShieldAlert className="h-14 w-14 text-sky-400 animate-pulse" />
              </div>
            </div>
            <div className="text-center">
              <p className="text-base font-semibold text-white">
                Distilling dataset…
              </p>
              <p className="mt-1 text-sm text-slate-400 max-w-md">
                Extracting universal features and running ensemble detection
                (Autoencoder + Deep&nbsp;SVDD + Isolation&nbsp;Forest)
              </p>
            </div>
          </div>
        )}

        {/* ── Results Dashboard ──────────────────────────────────── */}
        {results && !isScanning && (
          <div className="flex flex-col gap-8 animate-fade-up border-t border-[var(--color-border)] pt-8">
            
            {/* ── Action Header (Download) ─────────────────────────── */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-[var(--color-surface)] rounded-2xl p-6 border border-[var(--color-border)] shadow-xl relative overflow-hidden group">
              <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-cyan-500 via-sky-400 to-indigo-500 opacity-50"></div>
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5 text-emerald-400" />
                  Distillation Complete
                </h3>
                <p className="text-sm text-slate-400 mt-1">
                  Removed {results.poisoned_samples} anomalous items. Your data is ready.
                </p>
              </div>

              <button
                onClick={handleDownloadSanitized}
                disabled={isDownloading}
                className="relative inline-flex items-center gap-2 rounded-xl bg-slate-800 border border-slate-600 px-6 py-3 text-sm font-bold text-slate-200 shadow-md transition-all hover:text-cyan-300 hover:border-cyan-400 hover:shadow-[0_0_20px_rgba(34,211,238,0.4)] hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:pointer-events-none"
              >
                {isDownloading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Packaging Distilled Data...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 text-cyan-400" />
                    Download Sanitized Dataset
                  </>
                )}
              </button>
            </div>

            {/* ── Metric Cards ────────────────────────────────────── */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
              {/* Total Samples */}
              <MetricCard
                id="metric-total"
                icon={<Database className="h-5 w-5" />}
                label="Total Samples"
                value={results.total_samples.toLocaleString()}
                accent="sky"
              />
              {/* Poisoned Samples */}
              <MetricCard
                id="metric-poisoned"
                icon={<Bug className="h-5 w-5" />}
                label="Poisoned Samples"
                value={results.poisoned_samples.toLocaleString()}
                accent="rose"
              />
              {/* Anomaly Percentage */}
              <MetricCard
                id="metric-pct"
                icon={<Percent className="h-5 w-5" />}
                label="Anomaly Rate"
                value={`${results.anomaly_percentage}%`}
                accent={results.anomaly_percentage > 5 ? "rose" : "sky"}
                large
              />
            </div>

            {/* ── Charts Row: Donut + Bar + Flagged ────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              {/* ── Donut Chart ──────────────────────────────────── */}
              <div
                id="chart-card"
                className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6"
              >
                <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-sky-400" />
                  Data Integrity
                </h2>

                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="45%"
                      innerRadius={62}
                      outerRadius={96}
                      paddingAngle={4}
                      dataKey="value"
                      strokeWidth={0}
                      label={false}
                      labelLine={false}
                    >
                      <Cell fill={CHART_COLORS.clean} />
                      <Cell fill={CHART_COLORS.poisoned} />
                    </Pie>

                    {/* ── Custom SVG center text ── */}
                    <text
                      x="50%"
                      y="42%"
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="fill-cyan-300"
                      style={{ fontSize: "28px", fontWeight: 800 }}
                    >
                      {cleanPct}%
                    </text>
                    <text
                      x="50%"
                      y="52%"
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="fill-slate-500"
                      style={{ fontSize: "11px", fontWeight: 500 }}
                    >
                      Clean
                    </text>

                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        borderColor: "#334155",
                        borderRadius: "12px",
                        fontSize: "13px",
                        color: "#fff",
                      }}
                      itemStyle={{ color: "#e2e8f0" }}
                    />
                    <Legend
                      iconType="circle"
                      verticalAlign="bottom"
                      align="center"
                      wrapperStyle={{
                        fontSize: "12px",
                        color: "#94a3b8",
                        paddingTop: "12px",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* ── Model Comparison Bar Chart ────────────────────── */}
              <div
                id="model-breakdown"
                className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6"
              >
                <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-sky-400" />
                  Model Comparison
                </h2>

                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={barChartData}
                    layout="vertical"
                    margin={{ top: 0, right: 12, bottom: 0, left: 0 }}
                    barCategoryGap="28%"
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1e3a5f"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      allowDecimals={false}
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={110}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        border: "1px solid #334155",
                        borderRadius: "12px",
                        fontSize: "13px",
                        color: "#e2e8f0",
                      }}
                      cursor={{ fill: "#ffffff08" }}
                    />
                    <Bar
                      dataKey="flagged"
                      radius={[0, 6, 6, 0]}
                      maxBarSize={28}
                    >
                      {barChartData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {/* legend pills */}
                <div className="flex flex-wrap gap-2 mt-3 justify-center">
                  {barChartData.map((entry) => (
                    <span
                      key={entry.name}
                      className="inline-flex items-center gap-1.5 text-[11px] text-slate-400"
                    >
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: entry.fill }}
                      />
                      {entry.name}: {entry.flagged}
                    </span>
                  ))}
                </div>
              </div>

              {/* ── Flagged Items List ────────────────────────────── */}
              <div
                id="flagged-list"
                className="rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 flex flex-col"
              >
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-rose-400" />
                    Flagged Items
                    <span className="ml-1 rounded-full bg-rose-500/15 px-2 py-0.5 text-xs font-bold text-rose-400">
                      {results.flagged_items.length}
                    </span>
                  </h2>

                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-500" />
                    <input
                      id="filter-input"
                      type="text"
                      placeholder="Filter…"
                      value={filterQuery}
                      onChange={(e) => setFilterQuery(e.target.value)}
                      className="w-36 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-light)] pl-8 pr-3 py-1.5 text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
                    />
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto max-h-80 space-y-1.5 pr-1">
                  {filteredItems.length === 0 ? (
                    <p className="text-sm text-slate-600 text-center py-8">
                      {results.flagged_items.length === 0
                        ? "No anomalies detected — your dataset looks clean!"
                        : "No matches for this filter."}
                    </p>
                  ) : (
                    filteredItems.map((item, idx) => {
                      return (
                        <div
                          key={item.id}
                          onClick={() => setSelectedItem(item)}
                          className="flex items-center gap-2.5 rounded-lg bg-[var(--color-surface-light)] px-3 py-2.5 text-sm transition-colors cursor-pointer hover:bg-rose-500/10 group overflow-hidden"
                        >
                          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-rose-500/15 text-[9px] font-bold text-rose-400 group-hover:bg-rose-500/25">
                            {idx + 1}
                          </span>
                          <span className="font-mono text-slate-300 truncate text-xs flex-1">
                            {item.id}
                          </span>
                          <div className="flex items-center gap-1 shrink-0 flex-wrap justify-end">
                            {item.flagged_by.map((model) => {
                              const style = MODEL_COLORS[model] ?? DEFAULT_MODEL_STYLE;
                              return (
                                <span
                                  key={model}
                                  className={`rounded-full ${style.bg} px-1.5 py-0.5 text-[9px] font-semibold ${style.text} whitespace-nowrap`}
                                >
                                  {model}
                                </span>
                              );
                            })}
                            <button
                              className="ml-2 text-slate-500 hover:text-cyan-400 transition-colors"
                              title="Inspect Explanation"
                            >
                              <Search className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── XAI Modal ─────────────────────────────────────────────── */}
      {selectedItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm p-4">
          <div className="w-full max-w-3xl rounded-2xl border border-slate-700 bg-slate-800 shadow-2xl flex flex-col">
            <div className="flex items-center justify-between border-b border-slate-700 p-4">
              <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                Explainable AI: <span className="font-mono text-sm text-slate-400">{selectedItem.id}</span>
              </h3>
              <button 
                onClick={() => setSelectedItem(null)} 
                className="text-slate-400 hover:text-white transition-colors p-1"
              >
                ✕
              </button>
            </div>
            
            <div className="p-6 overflow-y-auto max-h-[80vh]">
              {selectedItem.explanation?.type === "image" && (
                <div className="w-full flex flex-col gap-4">
                   <h4 className="text-sm text-slate-400 text-center font-mono font-semibold uppercase tracking-wider mb-2">Latent Space Divergence</h4>
                   <p className="mt-0 mb-4 text-xs text-rose-300 bg-rose-500/10 p-3 rounded-lg border border-rose-500/20 text-center mx-auto shadow-inner leading-relaxed max-w-2xl">
                     Showing the specific ResNet-18 neural features that mathematically alienated this image from the clean dataset.
                   </p>
                   {selectedItem.explanation?.error ? (
                     <div className="flex flex-col items-center justify-center h-48 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 p-6 text-center">
                       <AlertTriangle className="w-8 h-8 mb-2 opacity-80" />
                       <p className="font-mono text-sm">{selectedItem.explanation.error}</p>
                     </div>
                   ) : (
                     <>
                       <div className="h-64 w-full pl-6">
                         <ResponsiveContainer width="100%" height="100%">
                           <BarChart data={selectedItem.explanation.top_contributors} layout="vertical" margin={{ left: 80, right: 30, top: 0, bottom: 0 }}>
                             <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                             <XAxis type="number" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                             <YAxis type="category" dataKey="feature_index" tickFormatter={(val) => `Dim ${val}`} stroke="#cbd5e1" fontSize={11} tickLine={false} axisLine={false} width={100} />
                             <Tooltip 
                                cursor={{fill: '#ffffff05'}}
                                contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", color: "#fff", borderRadius: "8px", border: "1px solid #475569" }}
                                formatter={(value: ValueType | undefined) => [Number(value ?? 0).toFixed(4), "Deviation"]}
                                labelFormatter={(label) => `Feature Dimension ${label}`}
                             />
                             <Bar dataKey="deviation_score" fill="#c084fc" radius={[0, 4, 4, 0]} maxBarSize={30}>
                               {selectedItem.explanation.top_contributors.map((_entry: ImageContributor, index: number) => (
                                  <Cell key={`cell-${index}`} fill={index < 3 ? "#c084fc" : "#a855f7"} />
                               ))}
                             </Bar>
                           </BarChart>
                         </ResponsiveContainer>
                       </div>
                       <p className="mt-2 text-xs text-rose-300 bg-rose-500/10 p-3 rounded-lg border border-rose-500/20 text-center mx-auto shadow-inner leading-relaxed">
                         Showing top 10 semantic dimensions that heavily deviate from the dataset's expected median.
                       </p>
                     </>
                   )}
                </div>
              )}

              {selectedItem.explanation?.type === "tabular" && (
                <div className="w-full flex flex-col gap-4">
                   <h4 className="text-sm text-slate-400 text-center font-mono font-semibold uppercase tracking-wider mb-2">Top Anomaly Contributors</h4>
                   <div className="h-64 w-full pl-6">
                     <ResponsiveContainer width="100%" height="100%">
                       <BarChart data={selectedItem.explanation.top_contributors} layout="vertical" margin={{ left: 80, right: 30, top: 0, bottom: 0 }}>
                         <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                         <XAxis type="number" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                         <YAxis type="category" dataKey="name" stroke="#cbd5e1" fontSize={11} tickLine={false} axisLine={false} width={100} />
                         <Tooltip 
                            cursor={{fill: '#ffffff05'}}
                            contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155", color: "#fff", borderRadius: "8px", border: "1px solid #475569" }}
                         />
                         <Bar dataKey="error" fill="#f43f5e" radius={[0, 4, 4, 0]} maxBarSize={30}>
                           {selectedItem.explanation.top_contributors.map((_entry: TabularContributor, index: number) => (
                              <Cell key={`cell-${index}`} fill={index === 0 ? "#f43f5e" : index === 1 ? "#fb7185" : "#fda4af"} />
                           ))}
                         </Bar>
                       </BarChart>
                     </ResponsiveContainer>
                   </div>
                   <p className="mt-2 text-xs text-slate-500 text-center mx-auto">
                     Showing the top 3 column features contributing to the absolute reconstruction error.
                   </p>
                </div>
              )}

              {!selectedItem.explanation && (
                <div className="text-slate-400 text-center py-12 flex flex-col items-center gap-3">
                  <Activity className="w-8 h-8 text-slate-600 animate-pulse" />
                  No explanation payload available for this item.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-[var(--color-border)] bg-[var(--color-surface)]/60 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between text-xs text-slate-600">
          <span>AICS Universal Sanitizer v0.2.0</span>
          <span>Autoencoder · Deep&nbsp;SVDD · Isolation&nbsp;Forest</span>
        </div>
      </footer>
    </main>
  );
}

/* ================================================================== */
/* Metric Card                                                        */
/* ================================================================== */
function MetricCard({
  id,
  icon,
  label,
  value,
  accent,
  large,
}: {
  id: string;
  icon: React.ReactNode;
  label: string;
  value: string;
  accent: "sky" | "rose";
  large?: boolean;
}) {
  const ring =
    accent === "sky"
      ? "ring-sky-500/20 text-sky-400"
      : "ring-rose-500/20 text-rose-400";
  const iconBg =
    accent === "sky" ? "bg-sky-500/10" : "bg-rose-500/10";
  const valColor =
    accent === "sky" ? "text-cyan-300" : "text-rose-300";

  return (
    <div
      id={id}
      className={`rounded-2xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6 ring-1 ${ring} transition-shadow hover:shadow-lg hover:shadow-sky-500/5`}
    >
      <div className="flex items-center gap-2 text-slate-400 mb-3">
        <span className={`rounded-lg p-1.5 ${iconBg}`}>{icon}</span>
        <span className="text-xs font-medium uppercase tracking-wider">
          {label}
        </span>
      </div>
      <p
        className={`${
          large ? "text-5xl" : "text-3xl"
        } font-extrabold tracking-tight ${valColor}`}
      >
        {value}
      </p>
    </div>
  );
}
