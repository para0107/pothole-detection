/**
 * frontend/src/pages/DamagePage.jsx — every recorded road fault, two ways.
 *
 * Replaces the old ExplorerPage and PriorityPage, which were the same entity
 * shown twice: "browse and audit everything" and "act on the worst first".
 * Keeping them apart meant two nav entries, two mental models and two places
 * to fix a bug, so they are now one destination with an explicit view switch:
 *
 *   Queue (default)  ranked repair plan from GET /priority-list, with cost
 *                    estimates, selection, the budget planner, and work-order
 *                    creation. This is the "what do we do on Monday" view.
 *   Table            server-filtered/sorted/paginated GET /detections with row
 *                    actions, bulk delete and CSV export. The audit view.
 *
 * The view lives in the query string (?view=table) so both are linkable and
 * the browser back button behaves. /priority and /explorer redirect here.
 *
 * Two features were dead on arrival in the old PriorityPage and are restored
 * here: BudgetPlanner was imported but never rendered, and the work-order
 * modal had state and a submit handler but no markup — so "Create one from the
 * Repairs page", which WorkOrdersPage tells operators to do, was impossible.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  ListOrdered, AlertTriangle, Printer, Banknote, MapPin, HardHat, ArrowRight,
  ChevronUp, ChevronDown, ChevronLeft, ChevronRight, X, Download, Copy, Check,
  Trash2, Wrench, RotateCcw, Table as TableIcon, LayoutList,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import {
  fetchPriority, fetchStats, createWorkOrder,
  fetchDetections, updateDetectionStatus, deleteDetectionsBulk, downloadCsv,
} from '../utils/api'
import { fmtRon, fmtDate } from '../utils/format'
import {
  CLASS_LABELS, SEVERITY, SEVERITY_COLORS, ALL_CLASSES, estimateRepairCost,
} from '../utils/constants'
import { SevBadge, ClassDot, SectionTitle, Spinner, CenterState, EmptyState, Kpi } from '../components/ui'
import BudgetPlanner from '../components/BudgetPlanner'
import { useAuth } from '../context/AuthContext'

/** POST /work-orders accepts at most 200 detection ids per order. */
const MAX_WO_ITEMS = 200
const PAGE_SIZE = 25

const today = () => new Date().toISOString().slice(0, 10)

/** FastAPI `detail` is a string for our errors and a list for validation ones. */
const readErr = (e, fallback) => {
  const d = e?.response?.data?.detail
  if (typeof d === 'string' && d) return d
  if (Array.isArray(d)) {
    const msgs = d.map(x => x?.msg).filter(Boolean)
    if (msgs.length) return msgs.join('. ')
  }
  if (d && typeof d === 'object' && d.msg) return String(d.msg)
  return e?.message || fallback
}

/* ── shell ───────────────────────────────────────────────────────────────── */

export default function DamagePage() {
  const [params, setParams] = useSearchParams()
  const view = params.get('view') === 'table' ? 'table' : 'queue'
  const setView = (v) => {
    const next = new URLSearchParams(params)
    if (v === 'queue') next.delete('view'); else next.set('view', v)
    setParams(next, { replace: true })
  }

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={view === 'table' ? styles.innerWide : styles.inner}>
        <SectionTitle
          overline="Road damage"
          title={view === 'queue' ? 'Repair planner' : 'Detection audit'}
          right={<ViewSwitch view={view} onChange={setView} />}
        />
        {view === 'queue' ? <QueueView /> : <TableView />}
      </div>
    </div>
  )
}

function ViewSwitch({ view, onChange }) {
  const opts = [
    { id: 'queue', label: 'Queue', icon: LayoutList, hint: 'Ranked by priority, with cost' },
    { id: 'table', label: 'Table', icon: TableIcon, hint: 'Filter, sort and audit every record' },
  ]
  return (
    <div style={styles.switch} role="tablist" aria-label="Damage view">
      {opts.map(o => {
        const Icon = o.icon
        const on = view === o.id
        return (
          <button
            key={o.id}
            role="tab"
            aria-selected={on}
            title={o.hint}
            className={`btn btn-sm${on ? ' btn-active' : ' btn-ghost'}`}
            style={{ border: on ? undefined : '1px solid transparent' }}
            onClick={() => onChange(o.id)}
          >
            <Icon size={13} /> {o.label}
          </button>
        )
      })}
    </div>
  )
}

/* ── queue view (was PriorityPage) ───────────────────────────────────────── */

function QueueView() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const { data, loading, error } = useApi(() => fetchPriority(100), [])
  const [selected, setSelected] = useState(new Set())

  // Stats feed the budget planner; the queue itself comes from useApi above.
  const [stats, setStats] = useState(null)
  useEffect(() => {
    let alive = true
    fetchStats().then(d => { if (alive) setStats(d) }).catch(() => {})
    return () => { alive = false }
  }, [])

  // Work-order modal
  const [woOpen, setWoOpen] = useState(false)
  const [woForm, setWoForm] = useState({
    title: '', crew_name: '', scheduled_for: '', due_date: '', cost_estimate_ron: '', notes: '',
  })
  const [woSaving, setWoSaving] = useState(false)
  const [woError, setWoError] = useState(null)

  const items = useMemo(
    () => (data?.items || []).map(it => ({ ...it, est_cost: estimateRepairCost(it.damage_type, it.severity) })),
    [data])

  const totalBacklogCost = useMemo(() => items.reduce((a, it) => a + it.est_cost, 0), [items])
  const selectedItems = items.filter(it => selected.has(it.id))
  const selectedCost = selectedItems.reduce((a, it) => a + it.est_cost, 0)
  const overCap = selectedItems.length > MAX_WO_ITEMS

  const toggle = (id) => setSelected(prev => {
    const next = new Set(prev)
    next.has(id) ? next.delete(id) : next.add(id)
    return next
  })
  const selectTopN = (n) => setSelected(new Set(items.slice(0, n).map(it => it.id)))

  const openWorkOrder = () => {
    setWoError(null)
    setWoForm({
      title: `Repairs, ${selectedItems.length} sites, ${today()}`,
      crew_name: '', scheduled_for: '', due_date: '',
      cost_estimate_ron: String(Math.round(selectedCost)),
      notes: '',
    })
    setWoOpen(true)
  }

  const submitWorkOrder = async (e) => {
    e.preventDefault()
    if (woSaving) return
    const title = woForm.title.trim()
    if (!title) { setWoError('Give the work order a title.'); return }
    if (selectedItems.length === 0) { setWoError('Select at least one site.'); return }
    if (overCap) {
      setWoError(`A work order takes at most ${MAX_WO_ITEMS} sites. Remove ${selectedItems.length - MAX_WO_ITEMS} of them.`)
      return
    }
    const raw = woForm.cost_estimate_ron.trim()
    const cost = raw === '' ? null : Number(raw)
    if (cost !== null && (!Number.isFinite(cost) || cost < 0)) {
      setWoError('The cost estimate has to be a number, 0 or more.')
      return
    }
    setWoSaving(true)
    setWoError(null)
    try {
      await createWorkOrder({
        title,
        detection_ids: selectedItems.map(it => it.id),
        crew_name: woForm.crew_name.trim() || null,
        scheduled_for: woForm.scheduled_for || null,
        due_date: woForm.due_date || null,
        cost_estimate_ron: cost,
        notes: woForm.notes.trim() || null,
      })
      setWoOpen(false)
      setSelected(new Set())
      // Land on the Orders tab, not the Operations default: after creating
      // something you should be looking at the thing you created.
      navigate('/operations?tab=orders')
    } catch (err) {
      setWoError(readErr(err, 'Could not create the work order.'))
      setWoSaving(false)
    }
  }

  const printWorkOrder = () => printQueue(selectedItems.length > 0 ? selectedItems : items.slice(0, 20), user)

  if (loading) return <CenterState><Spinner label="Ranking the repair queue…" /></CenterState>
  if (error) return <EmptyState icon={AlertTriangle} title="API unreachable" sub={error} />
  if (items.length === 0) {
    return (
      <EmptyState
        icon={ListOrdered}
        title="Nothing to repair (yet)"
        sub="The priority queue fills up as surveys are processed."
        action={<Link to="/ingest" className="btn btn-accent">Upload footage</Link>}
      />
    )
  }

  return (
    <>
      <div style={styles.kpiGrid}>
        <Kpi delay="delay-1" icon={ListOrdered} label="Queue size" value={items.length}
             sub="highest-priority open damage sites" />
        <Kpi delay="delay-2" icon={Banknote} label="Backlog estimate" value={fmtRon(totalBacklogCost)}
             sub="heuristic repair cost of the whole queue" color="var(--orange)" />
        <Kpi delay="delay-3" icon={HardHat} label="Selected for crew" value={selected.size}
             sub={selected.size ? `≈ ${fmtRon(selectedCost)}` : 'select rows to build a work order'} color="var(--cyan)" />
      </div>

      {/* Selection toolbar — the bridge from "ranked list" to "crew job". */}
      <div className="card anim-fade-up" style={styles.queueBar}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Select</span>
        <button className="btn btn-sm" onClick={() => selectTopN(10)}>Top 10</button>
        <button className="btn btn-sm" onClick={() => selectTopN(25)}>Top 25</button>
        {selected.size > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => setSelected(new Set())}>
            <X size={12} /> Clear
          </button>
        )}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="btn btn-sm" onClick={printWorkOrder}>
            <Printer size={13} /> Print sheet
          </button>
          <button className="btn btn-sm btn-accent" onClick={openWorkOrder} disabled={selected.size === 0}>
            <HardHat size={13} /> Create work order
          </button>
        </span>
      </div>

      {overCap && (
        <div className="card" style={styles.warn}>
          <AlertTriangle size={14} style={{ color: 'var(--orange)', flexShrink: 0 }} />
          A work order takes at most {MAX_WO_ITEMS} sites — {selectedItems.length} are selected.
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
        {items.map((it, idx) => {
          const isSel = selected.has(it.id)
          const sevColor = SEVERITY_COLORS[it.severity] || 'var(--text-muted)'
          return (
            <div
              key={it.id}
              className={`card pressable anim-fade-up delay-${Math.min(Math.floor(idx / 4) + 1, 6)}`}
              onClick={() => toggle(it.id)}
              role="button"
              aria-pressed={isSel}
              /* The raw score is an internal ranking number, not something a
                 crew acts on — it lives here rather than in the row. */
              title={`Priority score ${it.priority_score.toFixed(3)}`}
              style={{
                ...styles.row,
                borderColor: isSel ? 'var(--border-accent)' : 'var(--border)',
                background: isSel ? 'var(--accent-dim)' : 'var(--bg-card)',
              }}
            >
              {/* Rank is a figure, not a badge — a bordered chip here competed
                  with the class tile beside it for the same attention. */}
              <span className="mono" style={{
                ...styles.rank,
                color: idx < 3 ? 'var(--accent)' : 'var(--text-muted)',
              }}>
                {idx + 1}
              </span>

              <ClassDot cls={it.damage_type} size={34} />

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span className="display" style={styles.rowTitle}>
                    {CLASS_LABELS[it.damage_type] || it.damage_type}
                  </span>
                  <SevBadge s={it.severity} compact />
                </div>
                {/* SEVERITY[s].action, not SEVERITY_ACTIONS[s]: the latter
                    prefixes the band name, which the chip already says. */}
                <div style={styles.rowAction}>{SEVERITY[it.severity]?.action}</div>
                <div className="mono" style={styles.rowMeta}>
                  <MapPin size={9} style={{ flexShrink: 0 }} />
                  {it.latitude.toFixed(4)}, {it.longitude.toFixed(4)}
                  <span style={{ opacity: 0.5 }}>·</span>
                  seen {it.detection_count}×
                  <span style={{ opacity: 0.5 }}>·</span>
                  {fmtDate(it.last_detected)}
                </div>
              </div>

              <div style={styles.rowRight}>
                <span className="overline" style={{ fontSize: 9 }}>Est. repair</span>
                <span className="mono" style={{ ...styles.rowCost, color: sevColor }}>
                  {fmtRon(it.est_cost)}
                </span>
              </div>

              {/* A definite mark for "this one is in the order", so selection
                  never depends on a background tint alone. */}
              <span style={{
                ...styles.check,
                borderColor: isSel ? 'var(--accent)' : 'var(--border-bright)',
                background: isSel ? 'var(--accent)' : 'transparent',
                color: 'var(--accent-contrast)',
              }} aria-hidden="true">
                {isSel && <Check size={12} strokeWidth={3} />}
              </span>
            </div>
          )
        })}
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', marginTop: 22 }}>
        <Link to="/map" className="btn">See all of it on the map <ArrowRight size={13} /></Link>
      </div>

      {/* The budget planner: what would it cost to clear X% of each band. */}
      <div style={{ marginTop: 34 }}>
        <BudgetPlanner stats={stats} />
      </div>

      {woOpen && (
        <WorkOrderModal
          form={woForm} setForm={setWoForm} onClose={() => setWoOpen(false)}
          onSubmit={submitWorkOrder} saving={woSaving} error={woError}
          count={selectedItems.length} cost={selectedCost}
        />
      )}
    </>
  )
}

function WorkOrderModal({ form, setForm, onClose, onSubmit, saving, error, count, cost }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const set = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))

  return (
    <div style={styles.overlay} onClick={onClose} role="dialog" aria-modal="true" aria-label="Create work order">
      <form className="card" style={styles.modal} onClick={e => e.stopPropagation()} onSubmit={onSubmit}>
        <div style={styles.modalHead}>
          <div>
            <div className="overline" style={{ color: 'var(--accent)' }}>New work order</div>
            <h3 className="display" style={{ fontSize: 17, marginTop: 4 }}>
              {count} site{count === 1 ? '' : 's'} · ≈ {fmtRon(cost)}
            </h3>
          </div>
          <button type="button" className="btn btn-sm btn-ghost" onClick={onClose} aria-label="Close">
            <X size={14} />
          </button>
        </div>

        {error && <div style={styles.modalErr} role="alert">{error}</div>}

        <label style={styles.label}>Title
          <input className="input" style={styles.field} value={form.title} onChange={set('title')} required />
        </label>
        <label style={styles.label}>Crew
          <input className="input" style={styles.field} value={form.crew_name} onChange={set('crew_name')}
                 placeholder="optional" />
        </label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          <label style={styles.label}>Scheduled for
            <input className="input" style={styles.field} type="date" value={form.scheduled_for} onChange={set('scheduled_for')} />
          </label>
          <label style={styles.label}>Due
            <input className="input" style={styles.field} type="date" value={form.due_date} onChange={set('due_date')} />
          </label>
        </div>
        <label style={styles.label}>Cost estimate (RON)
          <input className="input" style={styles.field} inputMode="numeric"
                 value={form.cost_estimate_ron} onChange={set('cost_estimate_ron')} />
        </label>
        <label style={styles.label}>Notes
          <textarea className="input" style={{ ...styles.field, resize: 'vertical', fontFamily: 'var(--font-sans)' }}
                    rows={3} value={form.notes} onChange={set('notes')} placeholder="optional" />
        </label>

        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          <button type="button" className="btn" style={{ flex: 1 }} onClick={onClose}>Cancel</button>
          <button className="btn btn-accent" style={{ flex: 2 }} disabled={saving}>
            {saving ? 'Creating…' : 'Create work order'}
          </button>
        </div>
      </form>
    </div>
  )
}

/**
 * The crew sheet. This is a print document on white paper, so it deliberately
 * does NOT use the app's severity ramp: those colours are tuned to glow on
 * near-black and wash out to nothing on white. Same order, darker values.
 */
function printQueue(rows, user) {
  const PRINT_SEV = { 1: '#5f7a3a', 2: '#9a7413', 3: '#a15718', 4: '#a8391f', 5: '#7d2412' }
  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>RDDS — Work order</title>
<style>
  body { font-family:'Segoe UI',sans-serif; color:#10141c; margin:0; }
  .cover { background:#1a1a18; color:#e8825f; padding:44px 44px 30px; }
  .cover h1 { margin:0 0 6px; font-size:28px; }
  .cover p { margin:0; color:#a8a49a; font-size:13px; }
  .body { padding:32px 44px; }
  table { width:100%; border-collapse:collapse; }
  th { background:#1a1a18; color:#e8825f; text-align:left; font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; padding:10px 13px; }
  td { padding:9px 13px; border-bottom:1px solid #eef0f4; font-size:12.5px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; }
  .total { margin-top:18px; font-size:15px; font-weight:700; text-align:right; }
</style></head><body>
<div class="cover">
  <h1>Road repair work order${user?.city ? ` — ${user.city}` : ''}</h1>
  <p>RDDS priority queue · ${rows.length} sites · generated ${new Date().toLocaleString()}</p>
</div>
<div class="body">
<table>
<thead><tr><th>#</th><th>Damage</th><th>Severity</th><th>Priority</th><th>GPS</th><th>Seen</th><th>Est. cost</th></tr></thead>
<tbody>
${rows.map((it, i) => {
  const sc = PRINT_SEV[it.severity] || '#555'
  return `<tr>
    <td>${i + 1}</td>
    <td>${CLASS_LABELS[it.damage_type] || it.damage_type}</td>
    <td><span class="badge" style="background:${sc}1f;color:${sc}">S${it.severity}</span></td>
    <td style="font-family:monospace">${it.priority_score.toFixed(3)}</td>
    <td style="font-family:monospace;font-size:11px">${it.latitude.toFixed(5)}, ${it.longitude.toFixed(5)}</td>
    <td>${it.detection_count}×</td>
    <td style="font-family:monospace">${Math.round(it.est_cost).toLocaleString()} RON</td>
  </tr>`
}).join('')}
</tbody></table>
<div class="total">Estimated total: ${Math.round(rows.reduce((a, r) => a + r.est_cost, 0)).toLocaleString()} RON</div>
<p style="font-size:10.5px;color:#7a8296;margin-top:26px">
  Cost figures are heuristic planning estimates derived from damage class and severity — not a contractor quote.
</p>
</div>
<script>window.onload = () => window.print()</script>
</body></html>`
  const w = window.open('', '_blank')
  if (!w) return
  w.document.write(html)
  w.document.close()
}

/* ── table view (was ExplorerPage) ───────────────────────────────────────── */

function SortIcon({ col, sortCol, sortDir }) {
  if (sortCol !== col) return <ChevronUp size={11} style={{ opacity: 0.2 }} />
  return sortDir === 'desc'
    ? <ChevronDown size={11} style={{ color: 'var(--accent)' }} />
    : <ChevronUp size={11} style={{ color: 'var(--accent)' }} />
}

function TableView() {
  const navigate = useNavigate()

  const [page, setPage] = useState(1)
  const [damageType, setDamageType] = useState('')
  const [severityMin, setSeverityMin] = useState('')
  const [severityMax, setSeverityMax] = useState('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [sortCol, setSortCol] = useState('priority_score')
  const [sortDir, setSortDir] = useState('desc')

  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedIds, setSelectedIds] = useState([])
  const [deleteSurveyLog, setDeleteSurveyLog] = useState(false)
  const [copiedId, setCopiedId] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchDetections({
        page,
        page_size: PAGE_SIZE,
        ...(damageType && { damage_type: damageType }),
        ...(severityMin && { severity_min: Number(severityMin) }),
        ...(severityMax && { severity_max: Number(severityMax) }),
        ...(dateFrom && { date_from: dateFrom }),
        ...(dateTo && { date_to: dateTo }),
        sort_by: sortCol,
        sort_order: sortDir,
      })
      setData(result)
      setSelectedIds([])
    } catch (e) {
      setError(readErr(e, 'Could not load detections.'))
    } finally {
      setLoading(false)
    }
  }, [page, damageType, severityMin, severityMax, dateFrom, dateTo, sortCol, sortDir])

  useEffect(() => { load() }, [load])

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => (d === 'desc' ? 'asc' : 'desc'))
    else { setSortCol(col); setSortDir('desc') }
    setPage(1)
  }

  const resetFilters = () => {
    setDamageType(''); setSeverityMin(''); setSeverityMax('')
    setDateFrom(''); setDateTo(''); setPage(1)
  }

  const hasFilters = damageType || severityMin || severityMax || dateFrom || dateTo
  const items = data?.items || []
  const total = data?.total || 0
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))

  const allChecked = items.length > 0 && selectedIds.length === items.length
  const toggleAll = () => setSelectedIds(allChecked ? [] : items.map(d => d.id))
  const toggleOne = (id) =>
    setSelectedIds(prev => (prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]))

  const copyGps = (d) => {
    navigator.clipboard?.writeText(`${d.latitude.toFixed(6)}, ${d.longitude.toFixed(6)}`)
    setCopiedId(d.id)
    setTimeout(() => setCopiedId(null), 1200)
  }

  const bulkDelete = async () => {
    if (selectedIds.length === 0) return
    if (!window.confirm(`Delete ${selectedIds.length} detection(s)? This cannot be undone.`)) return
    setBusy(true)
    try {
      await deleteDetectionsBulk(selectedIds, deleteSurveyLog)
      await load()
    } catch (e) {
      alert(`Delete failed: ${readErr(e, 'unknown error')}`)
    } finally {
      setBusy(false)
    }
  }

  const toggleFixed = async (d) => {
    setBusy(true)
    try {
      const updated = await updateDetectionStatus(d.id, !d.is_fixed)
      setData(prev => ({
        ...prev,
        items: prev.items.map(x => (x.id === d.id ? { ...x, is_fixed: updated.is_fixed } : x)),
      }))
    } catch (e) {
      alert(`Update failed: ${readErr(e, 'unknown error')}`)
    } finally {
      setBusy(false)
    }
  }

  const headers = [
    { key: '_check', label: '', sortable: false, width: 34 },
    { key: 'damage_type', label: 'Type', sortable: true },
    { key: 'severity', label: 'Severity', sortable: true },
    { key: 'confidence', label: 'Conf', sortable: true },
    { key: 'priority_score', label: 'Priority', sortable: true },
    { key: 'detection_count', label: 'Seen', sortable: true },
    { key: 'latitude', label: 'GPS', sortable: true },
    { key: 'last_detected', label: 'Last seen', sortable: true },
    { key: '_status', label: 'Status', sortable: false },
    { key: '_actions', label: '', sortable: false, width: 120 },
  ]

  return (
    <>
      <div className="card anim-fade-up" style={styles.filterBar}>
        <select className="select" value={damageType} onChange={e => { setDamageType(e.target.value); setPage(1) }}>
          <option value="">All classes</option>
          {ALL_CLASSES.map(c => <option key={c} value={c}>{CLASS_LABELS[c]}</option>)}
        </select>
        <select className="select" value={severityMin} onChange={e => { setSeverityMin(e.target.value); setPage(1) }}>
          <option value="">Min severity</option>
          {[1, 2, 3, 4, 5].map(s => <option key={s} value={s}>S{s}+</option>)}
        </select>
        <select className="select" value={severityMax} onChange={e => { setSeverityMax(e.target.value); setPage(1) }}>
          <option value="">Max severity</option>
          {[1, 2, 3, 4, 5].map(s => <option key={s} value={s}>≤ S{s}</option>)}
        </select>
        <input className="input" type="date" value={dateFrom} onChange={e => { setDateFrom(e.target.value); setPage(1) }} title="From date" />
        <input className="input" type="date" value={dateTo} onChange={e => { setDateTo(e.target.value); setPage(1) }} title="To date" />
        {hasFilters && (
          <button className="btn btn-sm btn-ghost" onClick={resetFilters}><X size={12} /> Clear</button>
        )}
        <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
            <span className="mono" style={{ color: 'var(--accent)' }}>{total.toLocaleString()}</span> records
          </span>
          <button className="btn btn-sm" onClick={downloadCsv}><Download size={13} /> Export CSV</button>
        </span>
      </div>

      {selectedIds.length > 0 && (
        <div className="glass anim-fade-in" style={styles.bulkBar}>
          <span style={{ fontSize: 12.5 }}>
            <span className="mono" style={{ color: 'var(--accent)', fontWeight: 700 }}>{selectedIds.length}</span> selected
          </span>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11.5, color: 'var(--text-muted)', cursor: 'pointer' }}>
            <input type="checkbox" checked={deleteSurveyLog} onChange={e => setDeleteSurveyLog(e.target.checked)} />
            also delete affected survey-log rows
          </label>
          <button className="btn btn-sm btn-danger" onClick={bulkDelete} disabled={busy}>
            <Trash2 size={13} /> Delete selected
          </button>
        </div>
      )}

      <div className="card anim-fade-up delay-1" style={{ overflow: 'hidden' }}>
        {error ? (
          <EmptyState icon={AlertTriangle} title="Could not load detections" sub={error}
                      action={<button className="btn" onClick={load}>Retry</button>} />
        ) : loading && !data ? (
          <CenterState><Spinner label="Loading records…" /></CenterState>
        ) : items.length === 0 ? (
          <EmptyState icon={TableIcon} title="No records match"
                      sub={hasFilters ? 'Try clearing some filters.' : 'Upload a survey to create detections.'} />
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={styles.table}>
              <thead>
                <tr>
                  {headers.map(h => (
                    <th key={h.key}
                        style={{ ...styles.th, width: h.width, cursor: h.sortable ? 'pointer' : 'default' }}
                        onClick={h.sortable ? () => toggleSort(h.key) : undefined}>
                      {h.key === '_check' ? (
                        <input type="checkbox" checked={allChecked} onChange={toggleAll} aria-label="Select all rows" />
                      ) : (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                          {h.label}
                          {h.sortable && <SortIcon col={h.key} sortCol={sortCol} sortDir={sortDir} />}
                        </span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {items.map(d => (
                  <tr key={d.id} className="table-row-hover" style={{ opacity: d.is_fixed ? 0.55 : 1 }}>
                    <td style={styles.td}>
                      <input type="checkbox" checked={selectedIds.includes(d.id)} onChange={() => toggleOne(d.id)}
                             aria-label={`Select detection ${d.id}`} />
                    </td>
                    <td style={styles.td}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                        <ClassDot cls={d.damage_type} size={24} />
                        <span style={{ fontWeight: 600, fontSize: 12.5 }}>{CLASS_LABELS[d.damage_type] || d.damage_type}</span>
                      </span>
                    </td>
                    <td style={styles.td}><SevBadge s={d.severity} compact /></td>
                    <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>
                      {(d.confidence * 100).toFixed(0)}%
                    </td>
                    <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--accent)' }}>
                      {(d.priority_score || 0).toFixed(3)}
                    </td>
                    <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{d.detection_count}×</td>
                    <td style={{ ...styles.td, fontFamily: 'var(--font-mono)', fontSize: 11 }}>
                      {d.latitude.toFixed(4)}, {d.longitude.toFixed(4)}
                    </td>
                    <td style={{ ...styles.td, fontSize: 11.5, color: 'var(--text-dim)' }}>{fmtDate(d.last_detected)}</td>
                    <td style={styles.td}>
                      {d.is_fixed ? (
                        <span className="mono" style={{ color: 'var(--green)', fontSize: 11, fontWeight: 700 }}>REPAIRED</span>
                      ) : (
                        <span className="mono" style={{ color: SEVERITY_COLORS[d.severity] || 'var(--text-muted)', fontSize: 11, fontWeight: 700 }}>OPEN</span>
                      )}
                    </td>
                    <td style={{ ...styles.td, whiteSpace: 'nowrap' }}>
                      <span style={{ display: 'inline-flex', gap: 4 }}>
                        <IconBtn title="Show on map"
                                 onClick={() => navigate('/map', { state: { focus: { id: d.id, lat: d.latitude, lon: d.longitude } } })}>
                          <MapPin size={12} />
                        </IconBtn>
                        <IconBtn title="Copy GPS" onClick={() => copyGps(d)}>
                          {copiedId === d.id ? <Check size={12} style={{ color: 'var(--green)' }} /> : <Copy size={12} />}
                        </IconBtn>
                        <IconBtn title={d.is_fixed ? 'Reopen' : 'Mark repaired'} onClick={() => toggleFixed(d)} disabled={busy}>
                          {d.is_fixed ? <RotateCcw size={12} /> : <Wrench size={12} />}
                        </IconBtn>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {items.length > 0 && (
          <div style={styles.pager}>
            <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
              Page <span className="mono" style={{ color: 'var(--text)' }}>{page}</span> of <span className="mono">{pageCount}</span>
            </span>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="btn btn-sm" disabled={page <= 1 || loading} onClick={() => setPage(p => p - 1)}>
                <ChevronLeft size={13} /> Prev
              </button>
              <button className="btn btn-sm" disabled={page >= pageCount || loading} onClick={() => setPage(p => p + 1)}>
                Next <ChevronRight size={13} />
              </button>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

function IconBtn({ children, title, onClick, disabled }) {
  return (
    <button className="btn btn-sm btn-ghost"
            style={{ width: 26, height: 26, padding: 0, border: '1px solid var(--border)' }}
            title={title} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}

const styles = {
  page: { minHeight: '100%', paddingTop: 'calc(var(--nav-h) + 26px)', paddingBottom: 40 },
  inner: { maxWidth: 900, margin: '0 auto', padding: '0 26px' },
  innerWide: { maxWidth: 1160, margin: '0 auto', padding: '0 26px' },
  switch: { display: 'flex', gap: 4, padding: 3, borderRadius: 'var(--radius-lg)', background: 'var(--bg-card2)' },
  kpiGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14 },
  queueBar: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '10px 14px', marginTop: 14 },
  warn: {
    display: 'flex', alignItems: 'center', gap: 9, padding: '10px 14px', marginTop: 10,
    fontSize: 12.5, color: 'var(--text-dim)', borderColor: 'var(--orange)',
  },
  row: { display: 'flex', alignItems: 'center', gap: 14, padding: '13px 16px', cursor: 'pointer', userSelect: 'none' },
  rank: {
    width: 22, flexShrink: 0, textAlign: 'right',
    fontSize: 12.5, fontWeight: 600, fontVariantNumeric: 'tabular-nums',
  },
  // Hierarchy is weight + size + leading as a set, not size alone.
  rowTitle: { fontWeight: 700, fontSize: 14, letterSpacing: '-0.01em', lineHeight: 1.2 },
  rowAction: { fontSize: 12.5, color: 'var(--text-dim)', marginTop: 3, lineHeight: 1.4 },
  rowMeta: {
    display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap',
    fontSize: 10.5, color: 'var(--text-muted)', marginTop: 5,
  },
  rowRight: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2, flexShrink: 0 },
  rowCost: { fontSize: 14, fontWeight: 700, fontVariantNumeric: 'tabular-nums', letterSpacing: '-0.01em' },
  check: {
    width: 20, height: 20, borderRadius: 999, flexShrink: 0,
    display: 'grid', placeItems: 'center',
    border: '1.5px solid', transition: 'var(--transition)',
  },
  filterBar: { display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '12px 14px', marginBottom: 12 },
  bulkBar: { display: 'flex', alignItems: 'center', gap: 16, padding: '10px 16px', marginBottom: 12, borderColor: 'var(--border-accent)' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: {
    textAlign: 'left', padding: '11px 12px', fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 700,
    letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border-bright)', background: 'var(--bg-card2)', whiteSpace: 'nowrap', userSelect: 'none',
  },
  td: { padding: '9px 12px', fontSize: 12.5, borderBottom: '1px solid var(--border)', verticalAlign: 'middle' },
  pager: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px' },
  overlay: {
    position: 'fixed', inset: 0, zIndex: 1200, background: 'rgba(6, 5, 4, 0.72)', backdropFilter: 'blur(3px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
  },
  modal: { width: 'min(520px, 100%)', maxHeight: '86vh', overflowY: 'auto', padding: 22 },
  modalHead: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12, marginBottom: 16 },
  modalErr: {
    background: 'color-mix(in srgb, var(--red) 10%, transparent)',
    border: '1px solid color-mix(in srgb, var(--red) 35%, transparent)',
    color: 'var(--red)', borderRadius: 'var(--radius)', padding: '9px 12px', fontSize: 12.5, marginBottom: 12,
  },
  label: { display: 'block', fontSize: 11.5, fontWeight: 600, color: 'var(--text-dim)', marginBottom: 11 },
  field: { width: '100%', marginTop: 5, fontSize: 13 },
}
