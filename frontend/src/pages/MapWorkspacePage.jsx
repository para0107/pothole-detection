/**
 * frontend/src/pages/MapWorkspacePage.jsx — the city's one spatial surface.
 *
 * Detections and the Road Quality Index used to be two nav entries and two
 * full-screen maps, which meant an operator comparing "where is the damage"
 * against "which streets score worst" had to leave one map to open the other
 * and lost their viewport doing it. They are now one destination with a layer
 * switch; /quality redirects here.
 *
 * What this deliberately does NOT do is merge the two implementations. They
 * look alike but they are not: MapPage draws point detections from a single
 * fetch, while QualityPage re-fetches an aggregated grid for the current
 * viewport on every moveend — debounced, with a 413 too-large path, a cell-size
 * control and its own exports. Folding that second, movement-coupled lifecycle
 * into MapPage (already 1,350 lines with a zone-drawing state machine, heatmap
 * mode, declutter and landmark fly-to) buys the user nothing they can see and
 * costs the codebase the one file nobody wants to touch. Only one of the two is
 * mounted at a time, so there is never a second Leaflet instance in memory.
 */

import React from 'react'
import { useSearchParams } from 'react-router-dom'
import { MapPin, Gauge } from 'lucide-react'
import MapPage from './MapPage'
import QualityPage from './QualityPage'

const LAYERS = [
  { id: 'detections', label: 'Detections', icon: MapPin, hint: 'Every recorded fault, by severity' },
  { id: 'quality', label: 'Quality', icon: Gauge, hint: 'Road Quality Index, scored per grid square' },
]

export default function MapWorkspacePage() {
  const [params, setParams] = useSearchParams()
  const layer = params.get('layer') === 'quality' ? 'quality' : 'detections'

  const setLayer = (id) => {
    const next = new URLSearchParams(params)
    if (id === 'detections') next.delete('layer'); else next.set('layer', id)
    setParams(next, { replace: true })
  }

  return (
    <>
      {/* Top-centre is the one strip both maps leave empty: MapPage owns the
          left (notice, landmarks) and right (basemap, zone, heat, report),
          QualityPage owns the left panel and the bottom-right legend. */}
      <div className="glass glass-blur" style={styles.dock} role="tablist" aria-label="Map layer">
        {LAYERS.map(l => {
          const Icon = l.icon
          const on = layer === l.id
          return (
            <button
              key={l.id}
              role="tab"
              aria-selected={on}
              title={l.hint}
              className={`btn btn-sm${on ? ' btn-active' : ' btn-ghost'}`}
              style={{ border: on ? undefined : '1px solid transparent' }}
              onPointerDown={() => setLayer(l.id)}
            >
              <Icon size={13} /> {l.label}
            </button>
          )
        })}
      </div>

      {layer === 'quality' ? <QualityPage /> : <MapPage />}
    </>
  )
}

const styles = {
  dock: {
    position: 'fixed',
    top: 'calc(var(--nav-h) + 14px)',
    left: '50%',
    transform: 'translateX(-50%)',
    zIndex: 1200,
    display: 'flex',
    gap: 4,
    padding: 4,
    borderRadius: 999,
  },
}
