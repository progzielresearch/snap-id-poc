// SnapID-Modern.jsx
// Tailwind + framer-motion + lucide-react
import React, { useRef, useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload, Sparkles, Loader2, Download, Images, History, Trash2,
  Eye, X, Palette, FileText, ExternalLink, CreditCard, ShieldCheck, ChevronDown, ChevronDownCircle, Check
} from "lucide-react";
import crypto from 'crypto';

/* ---------------------------
   Load dynamic country data
   (Adjust the path if needed)
----------------------------*/
import COUNTRIES_DATA from "./countries_data.json";

/* ---------------------------
   Constants
----------------------------*/
const DPI = 300; // fixed (not shown in UI)
const DOC_LABEL = { passport: "Passport", visa: "Visa", driving_license: "Driving License" };

/* ---------------------------
   Helpers
----------------------------*/
const parseRgbString = (s) => {
  const m = /^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$/.exec(String(s || ""));
  if (!m) return null;
  const t = [+m[1], +m[2], +m[3]];
  return t.every(v => v >= 0 && v <= 255) ? t : null;
};
const tupleToCss = ([r, g, b]) => `rgb(${r}, ${g}, ${b})`;

// Replace the existing humanizeColorKey with this:
const humanizeColorKey = (key) => {
  if (!key) return "";
  // true if key ends with "-default" / "_default" / " default"
  const isDefault = /(?:^|[-_\s])default$/i.test(key);
  // strip the trailing default token (with hyphen/underscore/space)
  const base = key.replace(/(?:[-_\s])default$/i, "");
  const pretty = base.replace(/[-_]/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
  return isDefault ? `${pretty} (Default)` : pretty;
};


const getCountryList = () =>
  Object.entries(COUNTRIES_DATA).map(([code, obj]) => ({
    code,
    name: obj.country_name || code,
    flag: obj.country_flag || "",
  }));

const getDimsFor = (code, docType) => {
  const raw = COUNTRIES_DATA[code]?.[docType]; // e.g. "413,531"
  if (!raw) return null;
  const [w, h] = raw.split(",").map((n) => parseInt(n.trim(), 10));
  if (!Number.isFinite(w) || !Number.isFinite(h)) return null;
  return { w, h };
};

const getColorOptions = (code, docType) => {
  const colorObj = COUNTRIES_DATA[code]?.color?.[docType] || {};
  // turn into array with stable ordering: default first, then others alpha
  const entries = Object.entries(colorObj);
  const defaultIdx = entries.findIndex(([k]) => /-default$/.test(k));
  const arranged = [
    ...(defaultIdx >= 0 ? [entries[defaultIdx]] : []),
    ...entries.filter((_, i) => i !== defaultIdx).sort((a, b) => a[0].localeCompare(b[0])),
  ];
  return arranged.map(([key, rgbStr]) => ({
    key,
    label: humanizeColorKey(key),
    rgbStr,
    isDefault: /-default$/.test(key),
  }));
};

const getDefaultColor = (code, docType) => {
  const opts = getColorOptions(code, docType);
  const def = opts.find((o) => o.isDefault) || opts[0];
  return def?.rgbStr || "";
};

const colorLabelFromValue = (rgbStr, options) => {
  const hit = options.find((o) => o.rgbStr === rgbStr);
  return hit ? hit.label : "Custom RGB";
};

const formatDim = (px, unit) => {
  if (!px) return "—";
  const inch = px / DPI;
  return unit === "cm" ? `${(inch * 2.54).toFixed(1)} cm` : `${inch.toFixed(2)} in`;
};

/* ---------------------------
   Reusable controls
----------------------------*/
function StyledSelect({ value, onChange, children, ariaLabel, disabled }) {
  return (
    <div className="relative">
      <select
        aria-label={ariaLabel}
        value={value}
        onChange={onChange}
        disabled={disabled}
        className={`appearance-none pr-9 pl-3 py-2 rounded-xl bg-slate-800/70 text-slate-100 border border-slate-700
                   focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm
                   ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
    </div>
  );
}

/* Color select per-country-per-doc */
function ColorSelectDynamic({ valueRgbString, onChange, options }) {
  const currentTuple = parseRgbString(valueRgbString) || [140, 200, 232];
  const disabled = !options?.length;

  return (
    <div className="flex items-center gap-3">
      <Palette className="h-4 w-4 text-indigo-300 hidden sm:block" />
      <StyledSelect
        ariaLabel="Background color"
        value={valueRgbString || ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      >
        {!options.length && <option value="">Pick a country…</option>}
        {options.map((opt) => (
          <option key={opt.key} value={opt.rgbStr}>
            {opt.label}
          </option>
        ))}
      </StyledSelect>

      {/* Swatch */}
      <span
        className="inline-block w-5 h-5 rounded ring-1 ring-white/10"
        title={colorLabelFromValue(valueRgbString, options)}
        style={{ background: tupleToCss(currentTuple) }}
      />
    </div>
  );
}

/* Country listbox with flags + SEARCH (custom dropdown) */
function CountrySelect({ value, onChange, countries, fullWidth = false }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const buttonRef = useRef(null);

  const selected = value ? countries.find((c) => c.code === value) : null;

  // Close on outside click
  useEffect(() => {
    const onDoc = (e) => {
      if (!buttonRef.current) return;
      if (!buttonRef.current.parentElement?.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return countries;
    return countries.filter(
      (c) =>
        (c.name || "").toLowerCase().includes(q) ||
        (c.code || "").toLowerCase().includes(q)
    );
  }, [countries, query]);

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={() => setOpen((s) => !s)}
        className={`${fullWidth ? "w-full" : "w-[220px]"} inline-flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-slate-800/70 text-slate-100 border border-slate-700 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500`}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2 min-w-0">
          {selected?.flag && (
            <img
              src={selected.flag}
              alt=""
              className="h-4 w-6 object-cover rounded-sm ring-1 ring-white/10"
            />
          )}
          <span className="truncate">{selected ? selected.name : "Select…"}</span>
        </div>
        <ChevronDown
          className={`h-4 w-4 text-slate-400 transition ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className="absolute z-50 mt-2 w-[280px] rounded-xl border border-white/10 bg-slate-900/95 backdrop-blur shadow-xl"
            role="listbox"
          >
            {/* Sticky search bar */}
            <div className="sticky top-0 z-10 p-2 bg-slate-900/95 border-b border-white/10">
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search country…"
                className="w-full px-3 py-2 rounded-lg bg-slate-800/70 text-slate-100 border border-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
              />
            </div>

            {/* Results list */}
            <div className="max-h-64 overflow-auto p-1">
              {filtered.length ? (
                filtered.map((c) => (
                  <button
                    key={c.code}
                    onClick={() => {
                      onChange(c.code);
                      setOpen(false);
                      setQuery("");
                    }}
                    className={`w-full flex items-center gap-2 px-2 py-2 rounded-lg text-sm hover:bg-white/10 ${
                      c.code === value ? "bg-white/10" : ""
                    }`}
                    role="option"
                    aria-selected={c.code === value}
                  >
                    {c.flag && (
                      <img
                        src={c.flag}
                        alt=""
                        className="h-4 w-6 object-cover rounded-sm ring-1 ring-white/10"
                      />
                    )}
                    <span className="truncate flex-1 text-left">{c.name}</span>
                    {c.code === value && <Check className="h-4 w-4 text-indigo-400" />}
                  </button>
                ))
              ) : (
                <div className="px-3 py-3 text-xs text-slate-400">No matches</div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* Fancy AI-ish loader (vertical scanner fixed) */
function AiSynthAnimation({ caption }) {
  const wrapperRef = useRef(null);
  const [scanRange, setScanRange] = useState(0);
  const SCANNER_H = 96;

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const compute = () => {
      const h = el.clientHeight || 0;
      setScanRange(Math.max(0, h - SCANNER_H));
    };
    compute();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(compute) : null;
    if (ro) ro.observe(el);
    window.addEventListener("resize", compute);
    return () => { if (ro) ro.disconnect(); window.removeEventListener("resize", compute); };
  }, []);

  const nodes = Array.from({ length: 56 });

  return (
    <div ref={wrapperRef} className="relative w-full h-full flex items-center justify-center overflow-hidden">
      <motion.div
        className="absolute -inset-24 rounded-full blur-3xl bg-[radial-gradient(ellipse_at_center,rgba(99,102,241,0.25),transparent_60%)]"
        animate={{ rotate: 360 }}
        transition={{ duration: 24, repeat: Infinity, ease: "linear" }}
      />
      <motion.div
        className="absolute left-0 right-0 h-24 bg-gradient-to-b from-transparent via-indigo-500/15 to-transparent"
        style={{ top: 0 }}
        initial={{ y: 0 }}
        animate={{ y: [0, scanRange, 0] }}
        transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut" }}
      />
      <div className="grid grid-cols-8 gap-3 opacity-80">
        {nodes.map((_, i) => (
          <motion.span
            key={i}
            className="block w-2 h-2 rounded-full bg-indigo-300/40"
            animate={{ opacity: [0.2, 1, 0.2], scale: [1, 1.8, 1] }}
            transition={{ duration: 1.8, repeat: Infinity, delay: (i % 8) * 0.08 }}
          />
        ))}
      </div>
      <div className="absolute bottom-6 text-center">
        <Loader2 className="mx-auto h-6 w-6 animate-spin text-indigo-300 mb-2" />
        <p className="text-xs text-slate-300">{caption}</p>
      </div>
    </div>
  );
}

/* ---------------------------
   Main App
----------------------------*/
export default function App() {
  // Controls
  const [docType, setDocType] = useState("passport");
  const [country, setCountry] = useState("");
  const [bgRGB, setBgRGB] = useState("173,216,230"); // will be reset on country/doc change

  // Files / network
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // Output + history
  const [resultUrl, setResultUrl] = useState(""); // watermarked preview (data URL)
  const [finalUrl, setFinalUrl] = useState("");   // clean PNG (data URL)
  const [pdfUrl, setPdfUrl] = useState("");       // PDF (data URL)
  const [resultAR, setResultAR] = useState(null);
  const [activeRunId, setActiveRunId] = useState(null);
  const [history, setHistory] = useState([]);

  // PDF Preview modal
  const [showPdf, setShowPdf] = useState(false);
  const [pdfBlobUrl, setPdfBlobUrl] = useState("");

  // Payment modal
  const [showPay, setShowPay] = useState(false);
  const [payTarget, setPayTarget] = useState(null); // 'png' | 'pdf' | 'pdf-preview'
  const [payBusy, setPayBusy] = useState(false);

  // Units (cm | in) for dimension labels
  const [unit, setUnit] = useState("cm"); // 'cm' | 'in'

  // rotating status lines while loading
  const STATUS_LINES = [
    "Detecting landmarks…",
    "Aligning head & shoulders…",
    "Enhancing facial detail…",
    "Inpainting attire…",
    "Compositing background…",
    "Exporting outputs…",
  ];
  const [statusIdx, setStatusIdx] = useState(0);
  useEffect(() => {
    if (!isLoading) return;
    const id = setInterval(() => setStatusIdx((i) => (i + 1) % STATUS_LINES.length), 1100);
    return () => clearInterval(id);
  }, [isLoading]);

  const inputRef = useRef(null);

  const countriesArr = useMemo(() => getCountryList(), []);
  const allowedCountries = useMemo(
    () => countriesArr.filter((c) => !!COUNTRIES_DATA[c.code]?.[docType]),
    [countriesArr, docType]
  );

  // Resolve current width/height from selections (from JSON)
  const dims = useMemo(() => (country ? getDimsFor(country, docType) : null), [country, docType]);

  const colorOptions = useMemo(
    () => (country ? getColorOptions(country, docType) : []),
    [country, docType]
  );

  const colorLabel = useMemo(
    () => colorLabelFromValue(bgRGB, colorOptions),
    [bgRGB, colorOptions]
  );

  // Keep aspect ratio in sync
  useEffect(() => {
    setResultAR(dims ? dims.w / dims.h : null);
  }, [dims]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      history.forEach(h => { if (h.outputUrl?.startsWith("blob:")) URL.revokeObjectURL(h.outputUrl); });
      if (resultUrl?.startsWith("blob:")) URL.revokeObjectURL(resultUrl);
      if (pdfBlobUrl?.startsWith("blob:")) URL.revokeObjectURL(pdfBlobUrl);
    };
  }, [history, resultUrl, pdfBlobUrl]);

  // Helpers
  const onFilesChosen = (list) =>
    setFiles(Array.from(list || []).filter(f => f.type.startsWith("image/")).slice(0, 3));

  // --- On change: reset color to default for the selected country/doc ---
  const applyCountry = (code) => {
    setCountry(code);
    const d = getDimsFor(code, docType);
    setResultAR(d ? d.w / d.h : null);
    // RESET COLOR to default of this country/doc
    const def = getDefaultColor(code, docType);
    if (def) setBgRGB(def);

    setActiveRunId(null);
    setResultUrl(""); setFinalUrl(""); setPdfUrl("");
  };

  // (Also reset color on docType change to avoid cross-country mismatches)
  const applyDocType = (dt) => {
    setDocType(dt);
    const d = country ? getDimsFor(country, dt) : null;
    setResultAR(d ? d.w / d.h : null);
    if (country) {
      const def = getDefaultColor(country, dt);
      if (def) setBgRGB(def);
    }
    setActiveRunId(null);
    setResultUrl(""); setFinalUrl(""); setPdfUrl("");
  };

  // Format dimension labels
  const prettyW = dims ? formatDim(dims.w, unit) : "—";
  const prettyH = dims ? formatDim(dims.h, unit) : "—";

  // Actions
  const runFrontalize = async () => {
    setError("");
    setResultUrl(""); setFinalUrl(""); setPdfUrl("");

    if (!files.length) return setError("Please add 1–3 images.");
    if (!dims) return setError("Please select a country.");
    if (!parseRgbString(bgRGB)) return setError("Select a valid background color.");

    const fd = new FormData();
    fd.append("background_rgb", bgRGB.trim());           // backend expects "r,g,b"
    fd.append("canvas_width", String(dims.w));
    fd.append("canvas_height", String(dims.h));
    files.forEach(f => fd.append("files", f));

    setIsLoading(true);
    try {
      const res = await fetch("http://194.59.165.64:8001/frontalize", { method: "POST", body: fd });
      if (!res.ok) {
        let detail = "Server error";
        try { const j = await res.json(); detail = j.error?.message || JSON.stringify(j); } catch {}
        throw new Error(detail);
      }
      const j = await res.json();
      const wm = j.watermarked_image;
      const clean = j.final_image;
      const pdf = j.pdf;

      if (!wm || !clean) throw new Error("Server returned invalid image payload.");
      setResultUrl(wm); setFinalUrl(clean); setPdfUrl(pdf || "");

      const entry = {
        id: crypto.randomUUID(),
        time: new Date().toISOString(),
        bg: bgRGB.trim(),
        docType, country,
        w: dims.w, h: dims.h,
        inputs: files.map(f => f.name),
        outputUrl: wm, finalUrl: clean, pdfUrl: pdf || "",
      };
      setHistory(h => [entry, ...h]);
      setActiveRunId(entry.id);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setIsLoading(false);
      setStatusIdx(0);
    }
  };

  const onSelectHistory = (id) => {
    const h = history.find(x => x.id === id);
    if (!h) return;
    setActiveRunId(id);
    setResultUrl(h.outputUrl);
    setFinalUrl(h.finalUrl || "");
    setPdfUrl(h.pdfUrl || "");
    setBgRGB(h.bg);
    setDocType(h.docType);
    setCountry(h.country);
    setResultAR(h.w / h.h);
  };

  const clearHistory = () => {
    history.forEach(h => { if (h.outputUrl?.startsWith("blob:")) URL.revokeObjectURL(h.outputUrl); });
    setHistory([]); setActiveRunId(null);
  };

  // --- Download helpers (gated by dummy payment) ---
  const doDownloadPNG = () => {
    if (!finalUrl) return;
    const a = document.createElement("a");
    a.href = finalUrl;
    a.download = `frontalized_${Date.now()}.png`;
    a.click();
  };
  const doDownloadPDF = () => {
    if (!pdfUrl) return;
    const a = document.createElement("a");
    a.href = pdfUrl;
    a.download = `frontalized_${Date.now()}.pdf`;
    a.click();
  };

  const requestPayment = (target) => {
    setPayTarget(target); // 'png' | 'pdf' | 'pdf-preview'
    setShowPay(true);
  };

  const confirmPayment = async () => {
    setPayBusy(true);
    setTimeout(async () => {
      setPayBusy(false);
      setShowPay(false);
      if (payTarget === "png") doDownloadPNG();
      if (payTarget === "pdf") doDownloadPDF();
      if (payTarget === "pdf-preview") await openPdfPreviewUnlocked();
      setPayTarget(null);
    }, 900);
  };

  const openPdfPreviewUnlocked = async () => {
    if (!pdfUrl) return;
    try {
      if (pdfUrl.startsWith("data:application/pdf")) {
        const resp = await fetch(pdfUrl);
        const blob = await resp.blob();
        const bUrl = URL.createObjectURL(blob);
        setPdfBlobUrl(bUrl);
      } else {
        setPdfBlobUrl(pdfUrl);
      }
      setShowPdf(true);
    } catch {
      doDownloadPDF();
    }
  };

  // Mobile controls
  const [showMobileControls, setShowMobileControls] = useState(false);

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-950 via-indigo-950 to-black text-slate-100">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur border-b border-white/10 bg-black/20">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Sparkles className="h-6 w-6 text-indigo-400" />
            <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">Snap ID</h1>
            <span className="ml-2 hidden sm:inline text-xs sm:text-sm text-slate-400">Formal Image Generator</span>
          </div>

          {/* Desktop controls */}
          <div className="hidden md:flex items-center gap-4 text-xs text-slate-300">
            {/* Country with search + flags */}
            <div className="flex items-center gap-2">
              <span className="hidden lg:inline">Country:</span>
              <CountrySelect
                value={country}
                onChange={applyCountry}
                countries={allowedCountries}
              />
            </div>

            {/* Document */}
            <div className="flex items-center gap-2">
              <span className="hidden lg:inline">Document:</span>
              <StyledSelect
                ariaLabel="Document"
                value={docType}
                onChange={(e) => applyDocType(e.target.value)}
              >
                {Object.entries(DOC_LABEL).map(([k, label]) => (
                  <option key={k} value={k}>{label}</option>
                ))}
              </StyledSelect>
            </div>

            {/* Color (per-country-per-doc) */}
            <ColorSelectDynamic
              valueRgbString={bgRGB}
              onChange={setBgRGB}
              options={colorOptions}
            />
          </div>

          {/* Mobile: toggle */}
          <button
            onClick={() => setShowMobileControls(s => !s)}
            className="md:hidden inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border border-white/10 bg-slate-800/60 text-xs"
          >
            <ChevronDown className={`h-4 w-4 transition ${showMobileControls ? "rotate-180" : ""}`} />
            Options
          </button>
        </div>

        {/* Mobile controls panel */}
        <AnimatePresence>
          {showMobileControls && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="md:hidden border-t border-white/10 bg-black/30"
            >
              <div className="max-w-7xl mx-auto px-4 py-3 grid grid-cols-1 gap-3">
                <div className="flex flex-col gap-1">
                  <label className="text-xs text-slate-400">Country</label>
                  <CountrySelect
                    value={country}
                    onChange={applyCountry}
                    countries={allowedCountries}
                    fullWidth
                  />
                </div>

                <div className="flex flex-col gap-1">
                  <label className="text-xs text-slate-400">Document</label>
                  <StyledSelect
                    ariaLabel="Mobile Document"
                    value={docType}
                    onChange={(e) => applyDocType(e.target.value)}
                  >
                    {Object.entries(DOC_LABEL).map(([k, label]) => (
                      <option key={k} value={k}>{label}</option>
                    ))}
                  </StyledSelect>
                </div>

                <div className="flex flex-col gap-1">
                  <label className="text-xs text-slate-400">Background</label>
                  <ColorSelectDynamic
                    valueRgbString={bgRGB}
                    onChange={setBgRGB}
                    options={colorOptions}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-[1fr_330px] gap-6">
        {/* Workbench */}
        <section className="space-y-4">
          <div
            onDrop={(e)=>{e.preventDefault(); onFilesChosen(e.dataTransfer.files);}}
            onDragOver={(e) => e.preventDefault()}
            className="rounded-3xl border border-white/10 bg-white/5 p-6 md:p-8 flex flex-col md:flex-row gap-6"
          >
            <div className="flex-1">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-medium">Upload up to 3 angled photos</h2>
                <button onClick={() => setFiles([])} className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
                  <Trash2 className="h-4 w-4" /> Clear
                </button>
              </div>

              <div className="rounded-2xl border border-dashed border-white/15 bg-black/20 p-6 flex items-center justify-center text-center">
                <div>
                  <Upload className="mx-auto h-10 w-10 text-indigo-400" />
                  <p className="mt-2 text-sm text-slate-300">Drag & drop images here, or</p>
                  <button onClick={() => inputRef.current?.click()} className="mt-2 inline-flex items-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-sm font-medium">
                    Choose Files
                  </button>
                  <p className="mt-2 text-xs text-slate-400">JPEG/PNG, max 3 files</p>
                </div>
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(e) => onFilesChosen(e.target.files)}
                  className="hidden"
                />
              </div>

              {!!files.length && (
                <div className="mt-4 grid grid-cols-3 gap-3">
                  {files.map((f, i) => (
                    <div key={i} className="relative group">
                      <img src={URL.createObjectURL(f)} alt={f.name} className="h-28 w-full object-cover rounded-xl ring-1 ring-white/10" />
                      <div className="absolute inset-0 rounded-xl bg-black/0 group-hover:bg-black/30 transition" />
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-6 flex flex-wrap items-center gap-3">
                <button
                  onClick={runFrontalize}
                  disabled={isLoading || !files.length || !country}
                  className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 to-fuchsia-600 disabled:from-slate-700 disabled:to-slate-700 px-5 py-2.5 text-sm font-semibold shadow-lg shadow-indigo-900/30"
                >
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                  {isLoading ? "Processing" : "Generate Formal Image"}
                </button>

                {finalUrl && (
                  <button
                    onClick={() => requestPayment("png")}
                    className="relative inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:shadow-lg px-4 py-2 text-sm font-medium"
                  >
                    <ChevronDownCircle className="h-4 w-4 rotate-180" />
                    <span>Download PNG</span>
                  </button>
                )}

                {pdfUrl && (
                  <>
                    <button
                      onClick={() => requestPayment("pdf-preview")}
                      className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-500 hover:shadow-lg px-4 py-2 text-sm font-medium"
                    >
                      <Eye className="h-4 w-4" />
                      <span>Preview PDF</span>
                    </button>
                    <button
                      onClick={() => requestPayment("pdf")}
                      className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-fuchsia-500 to-pink-500 hover:shadow-lg px-4 py-2 text-sm font-medium"
                    >
                      <Download className="h-4 w-4" />
                      <span>Download PDF</span>
                    </button>
                  </>
                )}
              </div>

              {error && <p className="mt-3 text-sm text-rose-300">{error}</p>}
            </div>

            {/* Live Output + OUTSIDE dimension lines */}
            <div className="md:w-80 w-full">
              <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-gradient-to-br from-indigo-900/40 to-fuchsia-900/30 p-3">
                {/* Top bar: unit switch LEFT, right shows inputs (kept per your approved version) */}
                <div className="mb-2 flex items-center justify-between text-xs">
                  <div className="inline-flex items-center gap-1 rounded-xl bg-black/30 px-2 py-1 ring-1 ring-white/10">
                    <button
                      onClick={() => setUnit("cm")}
                      className={`px-2 py-0.5 rounded-lg ${unit==="cm"?"bg-indigo-600 text-white":"text-slate-300 hover:text-white"}`}
                    >
                      cm
                    </button>
                    <button
                      onClick={() => setUnit("in")}
                      className={`px-2 py-0.5 rounded-lg ${unit==="in"?"bg-indigo-600 text-white":"text-slate-300 hover:text-white"}`}
                    >
                      in
                    </button>
                  </div>
                  <span>{files.length}/3 inputs</span>
                </div>

                {/* Lines + preview laid out in a small grid */}
                <div className="grid grid-cols-[36px_1fr] grid-rows-[36px_1fr] gap-2">
                  {/* Horizontal line ABOVE image (row 1, col 2) */}
                  <div className="col-[2/3] row-[1/2] flex items-center justify-center">
                    <div className="relative h-px w-full bg-white/10">
                      <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 whitespace-nowrap leading-none text-[10px] font-medium px-2 py-0.5 rounded-full bg-black/40 text-slate-200/80 ring-1 ring-white/5 backdrop-blur-[1px]">
                        {prettyW}
                      </span>
                    </div>
                  </div>

                  {/* VERTICAL line LEFT of image (row 2, col 1) */}
                  <div className="col-[1/2] row-[2/3] self-stretch flex items-center justify-center">
                    <div className="relative w-px h-full bg-white/10">
                      <span className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 -rotate-90 whitespace-nowrap leading-none text-[10px] font-medium px-2 py-0.5 rounded-full bg-black/40 text-slate-200/80 ring-1 ring-white/5 backdrop-blur-[1px]">
                        {prettyH}
                      </span>
                    </div>
                  </div>

                  {/* IMAGE CONTAINER (row 2, col 2) */}
                  <div className="col-[2/3] row-[2/3]">
                    <div
                      className="rounded-xl flex items-center justify-center bg-black/30 ring-1 ring-white/10 overflow-hidden"
                      style={{ aspectRatio: resultAR ?? "3 / 4" }}
                    >
                      <AnimatePresence mode="wait">
                        {isLoading ? (
                          <motion.div
                            key="loader"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="relative w-full h-full"
                          >
                            <AiSynthAnimation caption={STATUS_LINES[statusIdx]} />
                          </motion.div>
                        ) : resultUrl ? (
                          <motion.img
                            key={resultUrl}
                            src={resultUrl}
                            alt="Result (watermarked preview)"
                            className="w-full h-full object-contain"
                            initial={{ opacity: 0, scale: 1.03, filter: "blur(8px)" }}
                            animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                            transition={{ duration: 0.6 }}
                            onLoad={(e) => {
                              if (!country)
                                setResultAR((e.currentTarget.naturalWidth || 1) / (e.currentTarget.naturalHeight || 1));
                            }}
                          />
                        ) : (
                          <div className="text-center text-slate-400 px-6">
                            <Images className="mx-auto h-10 w-10 mb-2 opacity-70" />
                            <p className="text-sm">Your generated image will appear here.</p>
                          </div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>
                </div>

                {/* Footer: Country • Document • Color Name */}
                <div className="mt-2 text-xs text-slate-300 flex items-center justify-between">
                  <span className="truncate">
                    {country ? (COUNTRIES_DATA[country]?.country_name || country) : "—"} • {DOC_LABEL[docType]} • {colorLabel}
                  </span>
                  <span className="rounded-full bg-black/40 text-slate-200/80 ring-1 ring-white/5 px-2 py-0.5 text-[10px] whitespace-nowrap">
                    {dims ? `${prettyW} × ${prettyH}` : "—"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* History */}
        <aside className="rounded-3xl border border-white/10 bg-white/5 p-4 lg:sticky lg:top-20 h-fit">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <History className="h-4 w-4 text-indigo-300" />
              <h3 className="text-sm font-semibold">This Session</h3>
            </div>
            {!!history.length && (
              <button onClick={clearHistory} className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1">
                <Trash2 className="h-3 w-3" /> Clear
              </button>
            )}
          </div>

          {!history.length ? (
            <p className="text-xs text-slate-400">No runs yet. Generate to see history here.</p>
          ) : (
            <div className="space-y-3 max-h-[70vh] overflow-auto pr-1">
              {history.map(h => (
                <button
                  key={h.id}
                  onClick={() => onSelectHistory(h.id)}
                  className={`w-full text-left rounded-2xl border p-2 transition ${
                    activeRunId === h.id ? "border-indigo-500 bg-indigo-500/10" : "border-white/10 hover:bg-white/10"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div className="relative w-14 h-14 rounded-lg overflow-hidden ring-1 ring-white/10">
                      <img src={h.outputUrl} alt="history" className="w-full h-full object-cover" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-xs text-slate-200 truncate">
                        {new Date(h.time).toLocaleTimeString()}
                      </div>
                      <div className="text-[10px] text-slate-400">
                        {(COUNTRIES_DATA[h.country]?.country_name || h.country)} • {DOC_LABEL[h.docType]} • {colorLabelFromValue(h.bg, getColorOptions(h.country, h.docType))}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </aside>
      </main>

      {/* PDF Preview Modal */}
      <AnimatePresence>
        {showPdf && (
          <motion.div
            key="pdf-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ y: 30, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 30, opacity: 0 }}
              className="relative w-full max-w-4xl h-[80vh] rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900 to-slate-950 p-4"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-slate-200">
                  <FileText className="h-5 w-5" />
                  <span className="text-sm font-medium">PDF Preview</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => { if (pdfUrl) window.open(pdfUrl, "_blank"); }}
                    className="inline-flex items-center gap-2 rounded-xl bg-white/10 hover:bg-white/20 px-3 py-1.5 text-xs"
                  >
                    <ExternalLink className="h-4 w-4" />
                    Open in new tab
                  </button>
                  <button
                    onClick={() => setShowPdf(false)}
                    className="inline-flex items-center gap-2 rounded-xl bg-white/10 hover:bg-white/20 px-3 py-1.5 text-xs"
                  >
                    <X className="h-4 w-4" /> Close
                  </button>
                </div>
              </div>

              <div className="w-full h-[calc(100%-44px)] rounded-xl overflow-hidden border border-white/10 bg-black/30">
                {pdfBlobUrl ? (
                  <iframe title="PDF Preview" src={pdfBlobUrl} className="w-full h-full" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-400 text-sm">
                    Loading preview…
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dummy Payment Modal (gates PNG/PDF download and PDF preview) */}
      <AnimatePresence>
        {showPay && (
          <motion.div
            key="pay-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ y: 24, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 24, opacity: 0 }}
              className="relative w-full max-w-lg rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-5"
            >
              <div className="flex items-center gap-2 mb-3">
                <ShieldCheck className="h-5 w-5 text-emerald-400" />
                <h3 className="text-sm font-semibold">Secure Checkout (Demo)</h3>
              </div>

              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex items-center justify-between">
                  <span>Action</span>
                  <span className="font-medium">
                    {payTarget === "png" && "Download HD PNG"}
                    {payTarget === "pdf" && "Download A4 PDF"}
                    {payTarget === "pdf-preview" && "Open PDF Preview"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Country / Doc / BG</span>
                  <span className="font-medium">
                    {(COUNTRIES_DATA[country]?.country_name || country || "—")} • {DOC_LABEL[docType]} • {colorLabel}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Watermark</span>
                  <span className="font-medium text-emerald-400">None (clean output)</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Total</span>
                  <span className="font-semibold">US$ 0.00</span>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-end gap-2">
                <button
                  onClick={() => { if (!payBusy) { setShowPay(false); setPayTarget(null); } }}
                  className="inline-flex items-center gap-2 rounded-xl bg-white/10 hover:bg-white/20 px-3 py-2 text-xs"
                  disabled={payBusy}
                >
                  <X className="h-4 w-4" />
                  Cancel
                </button>
                <button
                  onClick={confirmPayment}
                  disabled={payBusy}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:shadow-lg px-4 py-2 text-sm font-medium"
                >
                  {payBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CreditCard className="h-4 w-4" />}
                  {payBusy ? "Processing…" : "Pay (Demo)"}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <footer className="px-4 py-6 text-center text-xs text-slate-500">
        Built for Snap ID • Session-only history (clears on refresh)
      </footer>
    </div>
  );
}
