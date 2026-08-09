/**
 * frontend/src/pages/OperationsPage.jsx — the repair workflow, end to end.
 *
 * Replaces the separate /triage and /workorders destinations. They were never
 * two jobs: a citizen report is triaged into a real detection, detections are
 * grouped into a work order, and the order runs open → scheduled → in_progress
 * → repaired → verified. Splitting that across two nav entries made operators
 * bounce between pages to follow one thing through.
 *
 *   Inbox   citizen reports awaiting a promote/dismiss decision (TriagePage)
 *   Orders  the crew board and its status flow (WorkOrdersPage)
 *
 * Both are mounted with `embedded` so this layout owns the page padding and
 * the title while each keeps its own behaviour intact. The tab lives in the
 * query string (?tab=orders) so it is linkable and back works; /triage and
 * /workorders redirect here.
 */

import React from 'react'
import { useSearchParams } from 'react-router-dom'
import { Inbox, ClipboardList } from 'lucide-react'
import { SectionTitle } from '../components/ui'
import TriagePage from './TriagePage'
import WorkOrdersPage from './WorkOrdersPage'

const TABS = [
  { id: 'inbox', label: 'Inbox', icon: Inbox, hint: 'Citizen reports awaiting a decision' },
  { id: 'orders', label: 'Work orders', icon: ClipboardList, hint: 'Crew jobs, open through verified' },
]

export default function OperationsPage() {
  const [params, setParams] = useSearchParams()
  const tab = params.get('tab') === 'orders' ? 'orders' : 'inbox'

  const setTab = (id) => {
    const next = new URLSearchParams(params)
    if (id === 'inbox') next.delete('tab'); else next.set('tab', id)
    setParams(next, { replace: true })
  }

  return (
    <div style={styles.page} className="page-grid-bg">
      <div style={styles.inner}>
        <SectionTitle
          overline="Repair workflow"
          title="Operations"
          right={
            <div style={styles.switch} role="tablist" aria-label="Operations view">
              {TABS.map(t => {
                const Icon = t.icon
                const on = tab === t.id
                return (
                  <button
                    key={t.id}
                    role="tab"
                    aria-selected={on}
                    title={t.hint}
                    className={`btn btn-sm${on ? ' btn-active' : ' btn-ghost'}`}
                    style={{ border: on ? undefined : '1px solid transparent' }}
                    onClick={() => setTab(t.id)}
                  >
                    <Icon size={13} /> {t.label}
                  </button>
                )
              })}
            </div>
          }
        />

        <p style={styles.lede}>
          {tab === 'inbox'
            ? 'Reports that several drivers confirmed independently are the most reliable, so they are listed first. Promoting one turns it into a real detection that flows through the map, the queue and work orders.'
            : 'A work order groups nearby damage into one job for one crew, and moves from open to verified. An order cannot be verified while any of its sites has been seen again since it was signed off.'}
        </p>

        {tab === 'inbox' ? <TriagePage embedded /> : <WorkOrdersPage embedded />}
      </div>
    </div>
  )
}

const styles = {
  page: { minHeight: '100%', paddingTop: 'calc(var(--nav-h) + 26px)', paddingBottom: 40 },
  inner: { maxWidth: 1160, margin: '0 auto', padding: '0 26px' },
  switch: { display: 'flex', gap: 4, padding: 3, borderRadius: 'var(--radius-lg)', background: 'var(--bg-card2)' },
  lede: { fontSize: 13, color: 'var(--text-muted)', maxWidth: '76ch', margin: '-6px 0 20px' },
}
