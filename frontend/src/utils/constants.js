// ── Damage class colours ──────────────────────────────────────────────────
import {
  Flag, Search, ShieldCheck, BadgeCheck, Flame, Wrench, Building2, Moon,
} from 'lucide-react'

/**
 * Damage-class colours, tuned for the Ember dark ground.
 *
 * These are lifted off the old paper values: a hue that reads as "earth
 * orange" on warm paper turns to mud on near-black, and the chips in ui.jsx
 * render them at 16–28% alpha over a dark card, which costs more luminance
 * again. Every value here clears 4.5:1 on --bg. Keep them in step with the
 * status ramp in index.css.
 */
export const CLASS_COLORS = {
  longitudinal_crack:        '#7fb0d6',   // steel
  transverse_crack:          '#cf94b4',   // mauve
  alligator_crack:           '#e8a04f',   // earth orange
  repaired_crack:            '#93a862',   // olive
  pothole:                   '#e5674a',   // brick red
  pedestrian_crossing_blur:  '#b18ec7',   // violet
  lane_line_blur:            '#d9ab45',   // ochre
  manhole_cover:             '#55b3a8',   // teal
  patchy_road:               '#c294bb',   // plum
  rutting:                   '#a9a294',   // warm slate
}

export const CLASS_LABELS = {
  longitudinal_crack:        'Longitudinal Crack',
  transverse_crack:          'Transverse Crack',
  alligator_crack:           'Alligator Crack',
  repaired_crack:            'Repaired Crack',
  pothole:                   'Pothole',
  pedestrian_crossing_blur:  'Crossing Blur',
  lane_line_blur:            'Lane Blur',
  manhole_cover:             'Manhole Cover',
  patchy_road:               'Patchy Road',
  rutting:                   'Rutting',
}

export const CLASS_ICONS = {
  longitudinal_crack:        '〰',
  transverse_crack:          '═',
  alligator_crack:           '⬡',
  repaired_crack:            '✔',
  pothole:                   '⬤',
  pedestrian_crossing_blur:  '⛜',
  lane_line_blur:            '─',
  manhole_cover:             '◎',
  patchy_road:               '▦',
  rutting:                   '∿',
}

export const ALL_CLASSES = Object.keys(CLASS_LABELS)

// ── Severity ──────────────────────────────────────────────────────────────
/**
 * Severity S1→S5 — one row per band, and every other severity map in the app
 * is derived from it. Previously these were four parallel object literals that
 * had to be edited in step; the landing page then added a fifth copy. One
 * table means a band can never disagree with itself.
 *
 * Brightness rises with urgency, so on the dark ground the brightest thing on
 * the screen is always the worst thing on the road. This is the opposite of
 * the paper edition, where S5 was the darkest value. The colours mirror
 * --s1..--s5 in index.css; change both together.
 */
export const SEVERITY = {
  1: { name: 'Monitor',   color: '#93a862', action: 'Log it and re-inspect at the next survey.' },
  2: { name: 'Schedule',  color: '#d9ab45', action: 'Add it to the routine maintenance plan.' },
  3: { name: 'Priority',  color: '#e08a3c', action: 'Fit it into the current repair cycle.' },
  4: { name: 'Urgent',    color: '#e5674a', action: 'Send a crew within the week.' },
  5: { name: 'Emergency', color: '#f2553c', action: 'Close the lane and repair now.' },
}

/** The bands in order — for legends, scales and pickers. */
export const SEVERITY_BANDS = Object.entries(SEVERITY)
  .map(([s, v]) => ({ s: Number(s), ...v }))

const bySeverity = (pick) =>
  Object.fromEntries(Object.entries(SEVERITY).map(([s, v]) => [s, pick(v, s)]))

export const SEVERITY_COLORS  = bySeverity(v => v.color)
export const SEVERITY_NAMES   = bySeverity(v => v.name)
export const SEVERITY_LABELS  = bySeverity((v, s) => `S${s} · ${v.name}`)
export const SEVERITY_ACTIONS = bySeverity(v => `${v.name}: ${v.action.charAt(0).toLowerCase()}${v.action.slice(1)}`)
export const SEVERITY_SHORT   = bySeverity((v, s) => `S${s}`)

// ── Map defaults ──────────────────────────────────────────────────────────
// Maps open on the signed-in user's city (see hooks/useCityCenter.js).
// These are the LAST-RESORT fallbacks for the instant before the city
// resolves (or if a legacy account has no city): a country-level view, so
// nothing city-specific is ever hardcoded into the map.
export const DEFAULT_CENTER = [45.9432, 24.9668]   // country centroid fallback
export const DEFAULT_ZOOM   = 7
export const CITY_ZOOM      = 13                   // zoom used once a city resolves

// No landmark list lives here. The map's fly-to menu is served per city by
// GET /cities/landmarks (free OSM lookup, cached forever in city_landmarks),
// so a deployment for any city works with no code edits. A hardcoded list for
// one demo city used to sit here and made every other city look broken.

// ── Basemaps (all key-free) ───────────────────────────────────────────────
export const BASEMAPS = {
  dark: {
    label: 'Dark',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attr: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/">CARTO</a>',
  },
  voyager: {
    label: 'Streets',
    url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    attr: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/">CARTO</a>',
  },
  satellite: {
    label: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr: '© <a href="https://www.esri.com/">Esri</a> — World Imagery',
  },
}

export const TILE_URL  = BASEMAPS.dark.url
export const TILE_ATTR = BASEMAPS.dark.attr

// ── Pipeline stages (must match orchestrator + backend session.json) ─────
export const PIPELINE_STAGES = [
  { key: 'preprocessor',        label: 'Preprocessor',        sub: 'Frame extraction · GPS sync · Lighting' },
  { key: 'detector',            label: 'RT-DETR Detector',    sub: 'RT-DETR-L inference · Confidence filter' },
  { key: 'segmentor',           label: 'SAM Segmentor',       sub: 'SAM 2.1 Tiny · 4 geometry features' },
  { key: 'depth_estimator',     label: 'Depth Estimator',     sub: 'Monodepth2 · Relative disparity' },
  { key: 'severity_classifier', label: 'Severity Classifier', sub: 'Rule-based S1–S5 · Weighted multi-signal' },
  { key: 'deduplicator',        label: 'Deduplicator',        sub: 'DBSCAN · Haversine clustering' },
  { key: 'db_writer',           label: 'DB Writer',           sub: 'PostGIS upsert · Priority score update' },
]

// ── Repair planning heuristics (client-side estimates, RON) ──────────────
// Rough unit costs for a repair plan sketch — presentation aid, not a quote.
export const REPAIR_COST_RON = {
  pothole:                   950,
  alligator_crack:           1400,
  longitudinal_crack:        420,
  transverse_crack:          420,
  patchy_road:               1800,
  rutting:                   2200,
  repaired_crack:            120,
  manhole_cover:             600,
  lane_line_blur:            260,
  pedestrian_crossing_blur:  340,
}
export const SEVERITY_COST_FACTOR = { 1: 0.5, 2: 0.8, 3: 1.0, 4: 1.45, 5: 2.1 }

/** Estimated repair cost for one detection (client-side sketch, not a quote). */
export function estimateRepairCost(damageType, severity) {
  const base = REPAIR_COST_RON[damageType] ?? 700
  const factor = SEVERITY_COST_FACTOR[severity] ?? 1.0
  return Math.round(base * factor)
}

// ── Work orders (mirrors backend/models_work.py WO_STATUSES) ─────────────
export const WORK_ORDER_STATUSES = [
  'open', 'scheduled', 'in_progress', 'repaired', 'verified', 'cancelled',
]

export const WORK_ORDER_LABELS = {
  open:        'Open',
  scheduled:   'Scheduled',
  in_progress: 'In progress',
  repaired:    'Repaired',
  verified:    'Verified',
  cancelled:   'Cancelled',
}

export const WORK_ORDER_COLORS = {
  open:        '#6b93b8',
  scheduled:   '#d0a83f',
  in_progress: '#d67f34',
  repaired:    '#7ba05b',
  verified:    '#4f9a92',
  cancelled:   '#8a8578',
}

// The board shows the live flow; cancelled orders are reachable by filter.
export const WORK_ORDER_BOARD = ['open', 'scheduled', 'in_progress', 'repaired', 'verified']

// ── Badges (mirrors BADGES in backend/gamification.py) ───────────────────
export const BADGES = {
  first_report:   { label: 'First report',   icon: '🚩', description: 'Sent a first hazard report.' },
  confirmed_10:   { label: 'Road scout',     icon: '🔎', description: 'Ten of your reports were confirmed.' },
  confirmed_50:   { label: 'Road guardian',  icon: '🛡️', description: 'Fifty of your reports were confirmed.' },
  verified_first: { label: 'Triple checked', icon: '✅', description: 'A report of yours reached verified.' },
  streak_7:       { label: 'Week streak',    icon: '🔥', description: 'Reported on seven days in a row.' },
  fixed_1:        { label: 'Fixer',          icon: '🔧', description: 'A hazard you reported was repaired.' },
  fixed_5:        { label: 'City changer',   icon: '🏙️', description: 'Five hazards you reported were repaired.' },
  night_reporter: { label: 'Night watch',    icon: '🌙', description: 'Reported a hazard late at night.' },
}

export const ALL_BADGE_KEYS = Object.keys(BADGES)

// Drawn icons for each badge — no emoji. Keys mirror BADGES above.
export const BADGE_ICONS = {
  first_report:   Flag,
  confirmed_10:   Search,
  confirmed_50:   ShieldCheck,
  verified_first: BadgeCheck,
  streak_7:       Flame,
  fixed_1:        Wrench,
  fixed_5:        Building2,
  night_reporter: Moon,
}

// Plain-language names for the points ledger reasons.
export const POINTS_REASONS = {
  event_confirmed: 'Report confirmed by other drivers',
  event_verified:  'Report verified',
  event_fixed:     'Reported hazard was repaired',
  event_promoted:  'Report accepted as an official record',
}

// ── Road Quality Index bands (mirrors backend/routes/quality.py) ─────────
export const RQI_BANDS = {
  A: { label: 'Very good', color: '#7ba05b', min: 85 },
  B: { label: 'Good',      color: '#a9b45a', min: 70 },
  C: { label: 'Fair',      color: '#d0a83f', min: 50 },
  D: { label: 'Poor',      color: '#d67f34', min: 30 },
  E: { label: 'Very poor', color: '#c0492a', min: 0 },
}

// ── Live event validation states ─────────────────────────────────────────
export const LIVE_STATUS_LABELS = {
  unverified: 'Reported once',
  confirmed:  'Confirmed',
  verified:   'Verified',
}

export const LIVE_STATUS_COLORS = {
  unverified: '#8a8578',
  confirmed:  '#d0a83f',
  verified:   '#7ba05b',
}
