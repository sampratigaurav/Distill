"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
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
  | { type: "tabular"; top_contributors: TabularContributor[] }
  | { type: "text"; snippet: string; column: string };

interface FlaggedItem {
  id: string;
  confidence: number;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  flagged_by: string[];
  model_scores: Record<string, number | null>;
  explanation?: Explanation;
}

interface ScanResults {
  total_samples: number;
  poisoned_samples: number;
  anomaly_percentage: number;
  model_breakdown: Record<string, number>;
  confidence_distribution: Record<string, number>;
  flagged_items: FlaggedItem[];
  warning?: string;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MAX_FILE_SIZE = 1024 * 1024 * 1024; // 1 GB

/** Chart: clean = green, poisoned = red */
const CHART_COLORS = { clean: "#22c55e", poisoned: "#ef4444" };

/** Per-model colors — orange / red / zinc monochrome palette */
const MODEL_COLORS: Record<string, { bg: string; text: string; bar: string }> = {
  Autoencoder:              { bg: "bg-orange-500/10", text: "text-orange-500", bar: "#f97316" },
  "Deep SVDD":              { bg: "bg-red-500/10",    text: "text-red-500",    bar: "#ef4444" },
  "Isolation Forest":       { bg: "bg-zinc-500/10",   text: "text-zinc-300",   bar: "#d4d4d8" },
  "Statistical Pre-filter": { bg: "bg-yellow-500/10", text: "text-yellow-400", bar: "#eab308" },
};
const DEFAULT_MODEL_STYLE = { bg: "bg-red-500/10", text: "text-red-500", bar: "#ef4444" };

const SEVERITY_COLORS = {
    CRITICAL: { bg:"bg-red-500/20",    text:"text-red-400",    border:"border-red-500" },
    HIGH:     { bg:"bg-orange-500/20", text:"text-orange-400", border:"border-orange-500" },
    MEDIUM:   { bg:"bg-yellow-500/20", text:"text-yellow-400", border:"border-yellow-500" },
    LOW:      { bg:"bg-zinc-500/20",   text:"text-zinc-400",   border:"border-zinc-600" },
};

/** Tooltip style shared across all charts */
const TOOLTIP_STYLE = {
  backgroundColor: "#18181b", // zinc-900
  border: "1px solid #27272a", // zinc-800
  borderRadius: "0",
  fontSize: "12px",
  color: "#f4f4f5", // zinc-100
  fontFamily: "var(--font-geist-mono), monospace",
};

/** Stepped status messages shown during the processing phase */
const SCAN_MESSAGES = [
  "[EXTRACTING FEATURES...]",
  "[TRAINING MODELS...]",
  "[CALIBRATING THRESHOLDS...]",
  "[SCANNING CHUNKS...]",
  "[FINALIZING RESULTS...]",
] as const;

/* ------------------------------------------------------------------ */
/* Page Component                                                      */
/* ------------------------------------------------------------------ */
export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedItem, setSelectedItem] = useState<FlaggedItem | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanPhase, setScanPhase] = useState<string | null>(null);
  const [liveStats, setLiveStats] = useState<{chunk:number; total:number; poisoned:number} | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);
  const [results, setResults] = useState<ScanResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [filterQuery, setFilterQuery] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [textPrompt, setTextPrompt] = useState<string>("");

  /* ---- File handling -------------------------------------------- */
  const handleFile = useCallback((f: File | null) => {
    if (f && f.size > MAX_FILE_SIZE) {
      setFile(null);
      setResults(null);
      setError("File too large. Maximum limit is 1 GB.");
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

  /* ---- Step message cycler -------------------------------------- */
  useEffect(() => {
    if (scanPhase === "UPLOADING" || !scanPhase) return;
    setStepIndex(0);
    const interval = setInterval(() => {
      setStepIndex((prev) => (prev + 1) % SCAN_MESSAGES.length);
    }, 4000);
    return () => clearInterval(interval);
  }, [scanPhase]);

  /* ---- Scan API call -------------------------------------------- */
  const scanDataset = async () => {
    if (!file) return;
    setIsScanning(true);
    setScanPhase("UPLOADING");
    setLiveStats(null);
    setResults(null);
    setError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);
      if (textPrompt.trim()) {
        formData.append("text_prompt", textPrompt.trim());
      }

      const response = await new Promise<Response>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            setUploadProgress(Math.round((e.loaded / e.total) * 100));
          }
        });
        xhr.addEventListener('load', () => {
          resolve(new Response(xhr.response, {
            status: xhr.status,
            headers: {
              'Content-Type': xhr.getResponseHeader('Content-Type') || ''
            }
          }));
        });
        xhr.addEventListener('error', () => reject(new Error('Upload failed')));
        xhr.open('POST', `${API_URL}/scan-stream`);
        xhr.responseType = 'blob';
        xhr.send(formData);
      });
      
      if (!response.body) throw new Error("No readable stream available");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        
        // SSE frames are separated by double newline
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''  // last may be incomplete, keep it
        
        for (const frame of frames) {
          // A frame can have multiple lines (event:, data:, id:, etc)
          // We only care about lines starting with "data:"
          const dataLine = frame
            .split('\n')
            .find(line => line.startsWith('data:'))
          
          if (!dataLine) continue
          
          // Strip "data:" prefix and trim whitespace
          const jsonStr = dataLine.slice(5).trim()
          
          if (!jsonStr || jsonStr === '[DONE]') continue
          
          try {
            const event = JSON.parse(jsonStr)
            
            if (event.event === 'phase') {
              setScanPhase(event.data)
            } else if (event.event === 'progress') {
              setLiveStats({
                chunk: event.chunk,
                total: event.total_so_far,
                poisoned: event.poisoned_so_far,
              })
            } else if (event.event === 'complete') {
              setResults(event.result)
              setIsScanning(false)
              setScanPhase(null)
            } else if (event.event === 'error') {
              setError(event.detail)
              setIsScanning(false)
              setScanPhase(null)
            }
          } catch (e) {
            // Skip unparseable frames silently
            console.warn('SSE parse skip:', dataLine)
            continue
          }
        }
      }
    } catch (err: any) {
      setError(err.message ?? "Scan failed. Is the backend running?");
    } finally {
      setIsScanning(false);
      setScanPhase(null);
    }
  };

  /* ---- Download ------------------------------------------------- */
  const handleDownloadSanitized = async () => {
    if (!file || !results) return;
    setIsDownloading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("flagged_items", JSON.stringify(results.flagged_items));
      formData.append("scan_results_json", JSON.stringify(results));

      const response = await axios.post(`${API_URL}/download-sanitized`, formData);
      const { download_url, filename } = response.data;
      const link = document.createElement("a");
      const baseUrl = API_URL.endsWith("/") ? API_URL.slice(0, -1) : API_URL;
      link.href = `${baseUrl}${download_url}`;
      link.setAttribute("download", filename || "sanitized_data.zip");
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
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
        { name: "Clean", value: results.total_samples - results.poisoned_samples },
        { name: "Poisoned", value: results.poisoned_samples },
      ]
    : [];

  const cleanPct = results ? (100 - results.anomaly_percentage).toFixed(2) : null;

  const filteredItems = results
    ? results.flagged_items.filter((item) =>
        item.id.toLowerCase().includes(filterQuery.toLowerCase()) ||
        item.severity.toLowerCase().includes(filterQuery.toLowerCase())
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
    <main className="flex-1 flex flex-col min-h-screen bg-zinc-950 font-sans text-white">
      {/* ── Header ──────────────────────────────────────────────── */}
      <header className="border-b border-zinc-800 bg-zinc-900 sticky top-0 z-50">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center gap-3">
          {/* Logo mark — sharp square with orange accent */}
          <div className="border border-orange-500 bg-orange-500/10 p-2 rounded-none">
            <ShieldCheck className="h-5 w-5 text-orange-500" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-widest uppercase text-white font-mono">
              DISTILL
            </h1>
            <p className="text-[10px] text-zinc-500 tracking-wider uppercase font-mono">
              Universal Data Sanitization &amp; Poisoning Detection
            </p>
          </div>

          {/* status indicator */}
          <span className="ml-auto inline-flex items-center gap-1.5 border border-zinc-800 px-3 py-1 text-[10px] font-mono font-medium text-zinc-400 uppercase tracking-wider rounded-none">
            <span className="h-1.5 w-1.5 bg-green-500 animate-pulse rounded-none" />
            System Online
          </span>
        </div>
      </header>

      {/* ── Body ─────────────────────────────────────────────────── */}
      <div className="mx-auto w-full max-w-7xl px-6 py-10 flex flex-col gap-8">

        {/* ── Upload Zone ─────────────────────────────────────────── */}
        <section
          id="upload-zone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className="group relative cursor-pointer border border-dashed border-zinc-800 bg-zinc-900 hover:border-orange-500 transition-colors duration-0 p-10 text-center rounded-none"
        >
          <input
            ref={fileInputRef}
            id="file-input"
            type="file"
            accept=".csv,.zip,.parquet"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
          />

          {file ? (
            <div className="flex flex-col items-center gap-3">
              <FileText className="h-10 w-10 text-orange-500" />
              <p className="font-mono text-base font-semibold text-white">{file.name}</p>
              <p className="font-mono text-xs text-zinc-500">
                {(file.size / 1024).toFixed(1)} KB — click or drop to replace
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Upload className="h-10 w-10 text-zinc-600 group-hover:text-orange-500 transition-colors duration-0" />
              <p className="text-sm font-medium text-zinc-400">
                Drag &amp; drop your file here
              </p>
              <p className="font-mono text-xs text-zinc-600">
                <span className="text-orange-500">.csv</span> tabular data &nbsp;·&nbsp;{" "}
                <span className="text-orange-500">.zip</span> image archives &nbsp;·&nbsp;{" "}
                <span className="text-orange-500">.parquet</span>{" "}HuggingFace datasets
              </p>
            </div>
          )}
        </section>

        <div className="flex flex-col gap-1.5">
          <label className="font-mono text-[10px] text-zinc-500 
                            uppercase tracking-widest">
            Describe normal data{" "}
            <span className="text-zinc-700 normal-case tracking-normal">
              (optional — images only, uses CLIP zero-shot)
            </span>
          </label>
          <input
            type="text"
            placeholder='e.g. "a photo of a man" — leave blank for unsupervised mode'
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="border border-zinc-800 bg-zinc-950 px-3 py-2 
                       font-mono text-xs text-zinc-300 
                       placeholder-zinc-600 focus:outline-none 
                       focus:border-orange-500 rounded-none w-full"
          />
        </div>

        {/* ── Scan Button ────────────────────────────────────────── */}
        <div className="flex items-center justify-between gap-4">
          <button
            id="scan-button"
            disabled={!file || isScanning}
            onClick={scanDataset}
            className="inline-flex items-center gap-2 border border-orange-500 bg-orange-500 px-8 py-2.5 text-sm font-bold font-mono text-black uppercase tracking-widest transition-none hover:bg-orange-400 active:bg-orange-600 disabled:bg-zinc-800 disabled:border-zinc-700 disabled:text-zinc-600 disabled:pointer-events-none rounded-none"
          >
            <Activity className="h-4 w-4" />
            {scanPhase === "UPLOADING"
              ? "[ UPLOADING... ]"
              : scanPhase
              ? "[ PROCESSING... ]"
              : "SCAN DATASET"}
          </button>

          <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-zinc-600 uppercase tracking-wider">
            <ShieldCheck className="h-3 w-3" />
            Processed in memory · zero retention
          </span>
        </div>

        {/* ── Error ──────────────────────────────────────────────── */}
        {error && (
          <div
            id="error-banner"
            className="border border-red-500 bg-red-500/10 px-4 py-3 text-xs font-mono text-red-500 flex items-start gap-3 rounded-none"
          >
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* ── Warning ─────────────────────────────────────────────── */}
        {results?.warning && (
          <div className="border border-yellow-500 bg-yellow-500/10
            px-4 py-3 text-xs font-mono text-yellow-400
            flex items-start gap-3 rounded-none">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{results.warning}</span>
          </div>
        )}

        {/* ── Scanning State ─────────────────────────────────────── */}
        {isScanning && (
          <div className="border border-zinc-800 bg-zinc-900 px-6 py-10 flex flex-col items-center gap-6 rounded-none">
            {/* scanner box */}
            <div className="relative h-28 w-28 border border-zinc-800 bg-zinc-950 overflow-hidden rounded-none">
              <div className="absolute inset-x-0 h-px bg-orange-500 animate-scan-line opacity-80" />
              <div className="flex h-full items-center justify-center">
                <ShieldAlert className="h-10 w-10 text-orange-500" />
              </div>
            </div>

            {/* Phase text */}
            {scanPhase === "UPLOADING" ? (
              <div className="flex flex-col items-center gap-3 w-full max-w-xs">
                <p className="font-mono text-xs text-orange-500 flex items-center gap-2 uppercase tracking-widest">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  [UPLOADING PAYLOAD...]
                </p>
                <div className="w-full h-px bg-zinc-800 overflow-hidden">
                  <div
                    className="h-full bg-orange-500 transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="font-mono text-[10px] text-zinc-600">
                  {uploadProgress}% uploaded
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4 w-full max-w-xs">
                <p
                  key={scanPhase || stepIndex}
                  className="font-mono text-xs text-orange-500 text-center uppercase tracking-widest animate-fade-in"
                >
                  {scanPhase === "UPLOADING"       ? "[UPLOADING PAYLOAD...]"      :
                   scanPhase === "TRAINING_MODELS" ? "[TRAINING MODELS...]"        :
                   scanPhase === "CALIBRATING"     ? "[CALIBRATING THRESHOLDS...]" :
                   SCAN_MESSAGES[stepIndex]}
                </p>
                {/* step marker dots */}
                <div className="flex gap-1 border border-zinc-800 p-1">
                  {SCAN_MESSAGES.map((_, i) => (
                    <span
                      key={i}
                      className={`h-2 transition-none duration-0 ${
                        i === stepIndex
                          ? "w-8 bg-orange-500"
                          : "w-2 bg-zinc-800"
                      }`}
                    />
                  ))}
                </div>
                {liveStats && (
                  <div className="flex flex-col items-center gap-1">
                    <div className="font-mono text-xs text-zinc-400">
                      chunk {liveStats.chunk} &middot;{" "}
                      {liveStats.total.toLocaleString()} samples scanned
                    </div>
                    <div className="font-mono text-xs text-red-400">
                      {liveStats.poisoned} anomalies detected so far
                    </div>
                    <div className="w-48 h-1 bg-zinc-800 rounded-none overflow-hidden mt-1">
                      <div
                        className="h-full bg-orange-500 transition-all duration-500"
                        style={{
                          width: liveStats.poisoned > 0
                            ? `${Math.min(
                                (liveStats.poisoned / liveStats.total) * 100 * 10,
                                100
                              )}%`
                            : '0%'
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Results Dashboard ──────────────────────────────────── */}
        {results && !isScanning && (
          <div className="flex flex-col gap-6 border-t border-zinc-800 pt-8">

            {/* ── Action Header ──────────────────────────────────── */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border border-zinc-800 bg-zinc-900 p-5 rounded-none">
              <div>
                <h3 className="font-mono text-sm font-bold text-white uppercase tracking-widest flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-green-500" />
                  Distillation Complete
                </h3>
                <p className="font-mono text-xs text-zinc-500 mt-1">
                  {results.poisoned_samples} anomalies removed. Dataset safe for downstream ops.
                </p>
              </div>

              <button
                onClick={handleDownloadSanitized}
                disabled={isDownloading}
                className="inline-flex items-center gap-2 border border-zinc-700 bg-zinc-800 px-5 py-2.5 text-xs font-mono font-bold text-white uppercase tracking-widest transition-none hover:border-orange-500 hover:text-orange-500 active:bg-orange-500 active:text-black disabled:opacity-50 disabled:pointer-events-none rounded-none"
              >
                {isDownloading ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    PACKAGING...
                  </>
                ) : (
                  <>
                    <Download className="h-3.5 w-3.5" />
                    DOWNLOAD PAYLOAD
                  </>
                )}
              </button>
            </div>

            {/* ── Metric Cards ────────────────────────────────────── */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-px bg-zinc-800">
              <MetricCard
                id="metric-total"
                icon={<Database className="h-4 w-4" />}
                label="Total Samples"
                value={results.total_samples.toLocaleString()}
                accent="neutral"
              />
              <MetricCard
                id="metric-poisoned"
                icon={<Bug className="h-4 w-4" />}
                label="Poisoned Samples"
                value={results.poisoned_samples.toLocaleString()}
                accent="danger"
              />
              <MetricCard
                id="metric-pct"
                icon={<Percent className="h-4 w-4" />}
                label="Anomaly Rate"
                value={`${results.anomaly_percentage}%`}
                accent={results.anomaly_percentage > 5 ? "danger" : "neutral"}
                large
              />
            </div>

            {/* ── Charts Row ──────────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-px bg-zinc-800">

              {/* ── Donut Chart ───────────────────────────────────── */}
              <div
                id="chart-card"
                className="bg-zinc-900 p-5 rounded-none"
              >
                <h2 className="font-mono text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Activity className="h-3.5 w-3.5 text-orange-500" />
                  Data Integrity
                </h2>

                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="44%"
                      innerRadius={58}
                      outerRadius={90}
                      paddingAngle={2}
                      dataKey="value"
                      strokeWidth={0}
                      label={false}
                      labelLine={false}
                    >
                      <Cell fill={CHART_COLORS.clean} />
                      <Cell fill={CHART_COLORS.poisoned} />
                    </Pie>
                    <text
                      x="50%"
                      y="41%"
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{ fill: "#f4f4f5", fontSize: "26px", fontWeight: 800, fontFamily: "var(--font-geist-mono)" }}
                    >
                      {cleanPct}%
                    </text>
                    <text
                      x="50%"
                      y="51%"
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{ fill: "#71717a", fontSize: "10px", fontFamily: "var(--font-geist-mono)", letterSpacing: "0.1em" }}
                    >
                      CLEAN
                    </text>
                    <Tooltip contentStyle={TOOLTIP_STYLE} itemStyle={{ color: "#f4f4f5" }} cursor={false} />
                    <Legend
                      iconType="square"
                      iconSize={8}
                      verticalAlign="bottom"
                      align="center"
                      wrapperStyle={{ fontSize: "10px", color: "#71717a", paddingTop: "12px", fontFamily: "var(--font-geist-mono)", textTransform: "uppercase", letterSpacing: "0.08em" }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* ── Model Comparison ──────────────────────────────── */}
              <div
                id="model-breakdown"
                className="bg-zinc-900 p-5 rounded-none"
              >
                <h2 className="font-mono text-[10px] font-semibold text-zinc-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Activity className="h-3.5 w-3.5 text-orange-500" />
                  Model Comparison
                </h2>

                <ResponsiveContainer width="100%" height={240}>
                  <BarChart
                    data={barChartData}
                    layout="vertical"
                    margin={{ top: 0, right: 12, bottom: 0, left: 0 }}
                    barCategoryGap="30%"
                  >
                    <CartesianGrid strokeDasharray="2 4" stroke="#2a2a2e" horizontal={false} />
                    <XAxis
                      type="number"
                      allowDecimals={false}
                      tick={{ fill: "#52525b", fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={115}
                      tick={{ fill: "#a1a1aa", fontSize: 10, fontFamily: "var(--font-geist-mono)" }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#ffffff06" }} />
                    <Bar dataKey="flagged" radius={0} maxBarSize={24}>
                      {barChartData.map((entry, idx) => (
                        <Cell key={idx} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                <div className="flex flex-wrap gap-3 mt-2">
                  {barChartData.map((entry) => (
                    <span
                      key={entry.name}
                      className="inline-flex items-center gap-1.5 font-mono text-[10px] text-zinc-500 uppercase"
                    >
                      <span className="h-1.5 w-1.5 rounded-none" style={{ backgroundColor: entry.fill }} />
                      {entry.name}: {entry.flagged}
                    </span>
                  ))}
                </div>
              </div>

              {/* ── Flagged Items List ────────────────────────────── */}
              <div
                id="flagged-list"
                className="bg-zinc-900 p-5 flex flex-col rounded-none"
              >
                <div className="flex items-center justify-between mb-4">
                  <h2 className="font-mono text-[10px] font-semibold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                    <ShieldAlert className="h-3.5 w-3.5 text-red-500" />
                    Flagged Items
                    <span className="ml-1 border border-red-500 bg-red-500/10 px-1.5 py-0.5 text-[9px] font-bold font-mono text-red-500 rounded-none">
                      {results.flagged_items.length}
                    </span>
                  </h2>

                  <div className="relative">
                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-zinc-600" />
                    <input
                      id="filter-input"
                      type="text"
                      placeholder="filter..."
                      value={filterQuery}
                      onChange={(e) => setFilterQuery(e.target.value)}
                      className="w-32 border border-zinc-800 bg-zinc-950 pl-7 pr-3 py-1 font-mono text-[10px] text-zinc-400 placeholder-zinc-600 focus:outline-none focus:border-orange-500 rounded-none"
                    />
                  </div>
                </div>

                {results.confidence_distribution && (
                    <div className="flex gap-2 mb-3">
                        {Object.entries(results.confidence_distribution).map(([sev, count]) => 
                            count > 0 && (
                                <span key={sev} 
                                    className={`font-mono text-[8px] font-bold px-1.5 py-0.5 
                                    border ${SEVERITY_COLORS[sev as keyof typeof SEVERITY_COLORS].border}
                                    ${SEVERITY_COLORS[sev as keyof typeof SEVERITY_COLORS].bg}
                                    ${SEVERITY_COLORS[sev as keyof typeof SEVERITY_COLORS].text}`}>
                                    {sev}: {count}
                                </span>
                            )
                        )}
                    </div>
                )}

                <div className="flex-1 overflow-y-auto max-h-72 flex flex-col gap-px">
                  {filteredItems.length === 0 ? (
                    <p className="font-mono text-[10px] text-zinc-600 text-center py-8 uppercase tracking-wider">
                      {results.flagged_items.length === 0
                        ? "No anomalies detected — dataset is clean."
                        : "No matches for this filter."}
                    </p>
                  ) : (
                    filteredItems.map((item, idx) => (
                      <div
                        key={item.id}
                        onClick={() => setSelectedItem(item)}
                        className="flex items-center gap-2 bg-zinc-950 px-3 py-2 text-xs transition-none cursor-pointer hover:bg-orange-500/20 hover:border-l-2 hover:border-l-orange-500 group border-l-2 border-l-transparent rounded-none"
                      >
                        <span className="flex h-4 w-4 shrink-0 items-center justify-center border border-zinc-800 font-mono text-[8px] font-bold text-zinc-500 rounded-none">
                          {idx + 1}
                        </span>
                        <span className="font-mono text-zinc-400 truncate text-[10px] flex-1">
                          {item.id}
                        </span>
                        
                        {/* Confidence bar */}
                        <div className="flex items-center gap-1.5 shrink-0">
                            <div className="w-16 h-1 bg-zinc-800 rounded-none overflow-hidden">
                                <div 
                                    className="h-full transition-none"
                                    style={{ 
                                        width: `${(item.confidence * 100).toFixed(0)}%`,
                                        backgroundColor: item.severity === 'CRITICAL' ? '#ef4444' 
                                            : item.severity === 'HIGH' ? '#f97316'
                                            : item.severity === 'MEDIUM' ? '#eab308' 
                                            : '#71717a'
                                    }}
                                />
                            </div>
                            <span className={`font-mono text-[9px] font-bold ${SEVERITY_COLORS[item.severity].text}`}>
                                {(item.confidence * 100).toFixed(0)}%
                            </span>
                        </div>

                        <div className="flex items-center gap-1 shrink-0 flex-wrap justify-end">
                          {item.flagged_by.map((model) => {
                            const style = MODEL_COLORS[model] ?? DEFAULT_MODEL_STYLE;
                            return (
                              <span
                                key={model}
                                className={`border border-current/20 ${style.bg} px-1 py-px font-mono text-[8px] font-semibold ${style.text} uppercase tracking-wide rounded-none`}
                              >
                                {model}
                              </span>
                            );
                          })}
                          <button
                            className="ml-1 text-zinc-600 hover:text-orange-500 transition-none"
                            title="Inspect Explanation"
                          >
                            <Search className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── XAI Modal ─────────────────────────────────────────────── */}
      {selectedItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/90 p-4">
          <div className="w-full max-w-3xl border border-zinc-800 bg-zinc-900 flex flex-col rounded-none shadow-2xl">
            {/* modal header */}
            <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3 bg-zinc-950">
              <h3 className="font-mono text-xs font-bold text-white uppercase tracking-widest flex items-center gap-2">
                <Activity className="w-4 h-4 text-orange-500" />
                XAI Report ·{" "}
                <span className="text-zinc-500 font-normal">{selectedItem.id}</span>
              </h3>
              <button
                onClick={() => setSelectedItem(null)}
                className="font-mono text-xs text-zinc-500 hover:text-white transition-none px-2 py-1 hover:bg-zinc-800 border border-transparent hover:border-zinc-700 rounded-none"
              >
                [X] CLOSE
              </button>
            </div>

            <div className="p-6 overflow-y-auto max-h-[80vh]">
              <div className="grid grid-cols-3 gap-px bg-zinc-800 mb-4">
                  {Object.entries(selectedItem.model_scores).map(([model, score]) => (
                      score !== null && (
                          <div key={model} className="bg-zinc-900 p-3">
                              <p className="font-mono text-[9px] text-zinc-500 uppercase 
                                  tracking-wider mb-1">{model}</p>
                              <p className="font-mono text-lg font-bold text-white">
                                  {((score ?? 0) * 100).toFixed(1)}%
                              </p>
                              <div className="w-full h-0.5 bg-zinc-800 mt-1.5">
                                  <div className="h-full bg-orange-500" 
                                      style={{width:`${(score??0)*100}%`}}/>
                              </div>
                          </div>
                      )
                  ))}
              </div>
              
              {/* Image XAI */}
              {selectedItem.explanation?.type === "image" && (
                <div className="w-full flex flex-col gap-4">
                  <h4 className="font-mono text-[10px] text-zinc-500 uppercase tracking-widest border-l-2 border-orange-500 pl-2">
                    LATENT SPACE DIVERGENCE — TOP DEVIANT FEATURES
                  </h4>
                  <p className="font-mono text-[10px] text-zinc-400 border border-red-500/20 bg-red-500/5 px-3 py-2 rounded-none">
                    Specific ResNet-18 neural features that alienated this pattern.
                  </p>
                  {selectedItem.explanation?.error ? (
                    <div className="flex flex-col items-center justify-center h-40 border border-red-500/30 text-red-500 font-mono text-xs p-4 rounded-none bg-zinc-950">
                      <AlertTriangle className="w-6 h-6 mb-2 opacity-70" />
                      {selectedItem.explanation.error}
                    </div>
                  ) : (
                    <>
                      <div className="h-64 w-full pl-4 bg-zinc-950 border border-zinc-800 p-2">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={selectedItem.explanation.top_contributors}
                            layout="vertical"
                            margin={{ left: 80, right: 30, top: 0, bottom: 0 }}
                          >
                            <CartesianGrid strokeDasharray="2 4" stroke="#2a2a2e" horizontal={false} />
                            <XAxis type="number" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} style={{ fontFamily: "var(--font-geist-mono)" }} />
                            <YAxis type="category" dataKey="feature_index" tickFormatter={(v) => `Dim ${v}`} stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} width={90} style={{ fontFamily: "var(--font-geist-mono)" }} />
                            <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#ffffff04" }} formatter={(value: ValueType | undefined) => [Number(value ?? 0).toFixed(4), "Deviation"]} labelFormatter={(label) => `Feature Dim ${label}`} />
                            <Bar dataKey="deviation_score" fill="#f97316" radius={0} maxBarSize={28}>
                              {selectedItem.explanation.top_contributors.map((_: ImageContributor, index: number) => (
                                <Cell key={`cell-${index}`} fill={index < 3 ? "#f97316" : "#c2410c"} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Tabular XAI */}
              {selectedItem.explanation?.type === "tabular" && (
                <div className="w-full flex flex-col gap-4">
                  <h4 className="font-mono text-[10px] text-zinc-500 uppercase tracking-widest border-l-2 border-red-500 pl-2">
                    RECONSTRUCTION ERROR — TOP ANOMALY CONTRIBUTORS
                  </h4>
                  <div className="h-64 w-full pl-4 bg-zinc-950 border border-zinc-800 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={selectedItem.explanation.top_contributors}
                        layout="vertical"
                        margin={{ left: 80, right: 30, top: 0, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="2 4" stroke="#2a2a2e" horizontal={false} />
                        <XAxis type="number" stroke="#52525b" fontSize={10} tickLine={false} axisLine={false} style={{ fontFamily: "var(--font-geist-mono)" }} />
                        <YAxis type="category" dataKey="name" stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} width={90} style={{ fontFamily: "var(--font-geist-mono)" }} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "#ffffff04" }} />
                        <Bar dataKey="error" fill="#ef4444" radius={0} maxBarSize={28}>
                          {selectedItem.explanation.top_contributors.map((_: TabularContributor, index: number) => (
                            <Cell key={`cell-${index}`} fill={index === 0 ? "#ef4444" : index === 1 ? "#b91c1c" : "#7f1d1d"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="font-mono text-[10px] text-zinc-600 text-center uppercase tracking-wider">
                    Top 3 column features by absolute reconstruction error
                  </p>
                </div>
              )}

              {/* Text / NLP XAI */}
              {selectedItem.explanation?.type === "text" && (
                <div className="w-full flex flex-col gap-4">
                  <h4 className="font-mono text-[10px] text-zinc-500 uppercase tracking-widest border-l-2 border-orange-500 pl-2">
                    SEMANTIC DEVIATION — NLP ANOMALY
                  </h4>
                  <p className="font-mono text-[10px] text-zinc-400 border border-orange-500/20 bg-orange-500/5 px-3 py-2 rounded-none">
                    Text snippet violated expected semantic topography.
                  </p>
                  <div className="relative border border-zinc-800 bg-zinc-950 p-5 max-w-2xl w-full rounded-none">
                    <span className="absolute -top-px left-4 bg-zinc-900 border border-zinc-800 border-t-0 font-mono text-[9px] font-bold text-orange-500 px-2 uppercase tracking-widest rounded-none">
                      col: {selectedItem.explanation.column}
                    </span>
                    <blockquote className="font-mono text-sm text-zinc-400 whitespace-pre-wrap border-l-2 border-orange-500 pl-4 py-1 italic bg-zinc-900/50 p-2">
                      &ldquo;{selectedItem.explanation.snippet}&rdquo;
                    </blockquote>
                  </div>
                </div>
              )}

              {!selectedItem.explanation && (
                <div className="font-mono text-[10px] text-zinc-600 text-center py-10 uppercase tracking-widest border border-dashed border-zinc-800 bg-zinc-950">
                  No explanation payload available for this item.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ────────────────────────────────────────────────── */}
      <footer className="mt-auto border-t border-zinc-800 bg-zinc-950">
        <div className="mx-auto max-w-7xl px-6 py-3 flex items-center justify-between font-mono text-[10px] text-zinc-600 uppercase tracking-widest">
          <span>Distill v0.2.0</span>
          <span>Autoencoder · Deep SVDD · Isolation Forest</span>
        </div>
      </footer>

      {/* ── API Access ─────────────────────────────────────────────── */}
      <div className="border-t border-zinc-800 bg-zinc-900 px-6 py-8 mx-auto w-full max-w-7xl">
        <h2 className="font-mono text-xs font-bold text-zinc-500 uppercase tracking-widest mb-4">
          API ACCESS
        </h2>
        <pre className="font-mono text-[10px] text-zinc-400 bg-zinc-950 border border-zinc-800 p-4 overflow-x-auto">
{`# Scan a dataset programmatically
curl -X POST ${API_URL}/scan-dataset \\
  -F "file=@your_dataset.csv" \\
  -H "X-API-Key: your_key_here"

# Parquet files also supported
curl -X POST ${API_URL}/scan-dataset \\
  -F "file=@dataset.parquet"`}
        </pre>
      </div>
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
  accent: "neutral" | "danger";
  large?: boolean;
}) {
  const valColor = accent === "danger" ? "text-red-500" : "text-white";
  const iconColor = accent === "danger" ? "text-red-500" : "text-zinc-500";

  return (
    <div
      id={id}
      className="bg-zinc-900 p-5 hover:bg-zinc-800 transition-none rounded-none"
    >
      <div className={`flex items-center gap-1.5 mb-3 ${iconColor}`}>
        {icon}
        <span className="font-mono text-[10px] font-medium uppercase tracking-widest text-zinc-500">
          {label}
        </span>
      </div>
      <p className={`font-mono font-extrabold tracking-tight ${valColor} ${large ? "text-5xl" : "text-3xl"}`}>
        {value}
      </p>
    </div>
  );
}
