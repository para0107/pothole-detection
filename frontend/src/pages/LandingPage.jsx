/**
 * frontend/src/pages/LandingPage.jsx — the public front door.
 *
 * Shown at "/" to logged-out visitors (signed-in users get the Command
 * dashboard instead). The whole product packaged around one promise — a safer
 * drive for the whole city — with road-damage detection as the mechanism.
 *
 * Design: "Ember", the same system the rest of the app runs on. This page
 * defines layout and type scale under .rdds-landing and takes every colour
 * from the tokens in index.css, so the marketing surface and the console
 * cannot drift apart. The app navbar rides transparently over the hero and
 * solidifies on scroll (see .nav-transparent + Navbar's heroTop), which is why
 * there is no second header here.
 *
 * Live data comes from GET /api/public/landing (stats, recent activity, a road
 * quality grade) and degrades to product facts when the city has no data yet.
 * "Request a pilot" posts to the defended /contact/sales capture.
 */

import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchLanding, requestPilot } from '../utils/api'
import { fmtNum, fmtDate } from '../utils/format'
import { SEVERITY, SEVERITY_BANDS, CLASS_LABELS } from '../utils/constants'
import {
  createSpring, project, rubberband, trackVelocity, handoff, nearest, clamp,
  useReveal, useReducedMotion, useParallax,
} from '../utils/motion'

const label = (t) => CLASS_LABELS[t] || (t ? t.replace(/_/g, ' ') : 'Damage')

const LOOP = [
  { n: '1', k: 'Detect', t: 'RT-DETR · SAM 2.1',
    d: 'Ten kinds of damage read straight from dashcam video, frame by frame: cracks, potholes, rutting, worn markings.' },
  { n: '2', k: 'Score', t: 'depth · area · contrast',
    d: 'A transparent S1 to S5 severity from measured signals, combined by a published rule. Not a black box that guesses.' },
  { n: '3', k: 'Prioritise', t: 'severity · ln(count+1)',
    d: 'A ranked queue your budget can work through. The most dangerous roads first, with how often a fault recurs weighted in.' },
  { n: '4', k: 'Repair', t: 'work orders · route',
    d: "Group faults into work orders, plan the crew's route on the map, and send them out with the evidence in hand." },
  { n: '5', k: 'Verify', t: 'reopened-damage guard',
    d: 'A repair is not done until the road stops showing the damage. If the fault comes back, so does the work order.' },
]

const SAFETY = [
  { k: 'For drivers',
    p: 'No blown tyre, no sudden swerve into the next lane. The worst potholes are found and ranked before they find you.' },
  { k: 'For cyclists and pedestrians',
    p: 'The edge breaks and sunken drains that put a cyclist on the ground are exactly the faults RDDS is built to catch first.' },
  { k: 'For the whole city',
    p: 'Crews spend the budget on the roads that actually endanger people, and the repair is proven to hold. Safety you can audit.' },
]

const REASONS = [
  { d: '01', h: 'Compute rides with the fleet',
    p: 'Every vehicle runs its own detection on its own hardware. The server only stores and ranks, so a bigger survey costs one spatial query, not a bigger bill.' },
  { d: '02', h: 'No paid AI, anywhere',
    p: 'Open models, free map tiles, a self-hosted check for abuse. There is no metered API hiding behind the product.' },
  { d: '03', h: 'Location is all we keep',
    p: 'A detection is a coordinate and a class. No street profiles, no driver identities, nothing resold to anyone.' },
  { d: '04', h: 'The record stays auditable',
    p: 'Every severity score follows a rule you can inspect, and a repair only closes when the road agrees it is fixed.' },
]

/** Live figures when the city has data, honest product facts when it does not. */
function statTiles(data) {
  const st = data?.stats
  const q = data?.quality
  if (st && st.total_detections > 0) {
    return {
      live: true,
      tiles: [
        { k: st.total_detections, l: 'Road faults mapped and on record' },
        { k: st.critical_count, l: 'Urgent hazards flagged (S4–S5)' },
        { k: st.fixed_count, l: 'Repairs verified as fixed' },
        { k: q ? q.grade : '—', l: 'Road quality grade, updated live', plain: true },
      ],
    }
  }
  return {
    live: false,
    tiles: [
      { k: 10, u: 'classes', l: 'Damage types on the N-RDD2024 schema' },
      { k: 'S1–S5', plain: true, l: 'Transparent severity, a readable rule' },
      { k: '0.00', u: '/report', plain: true, l: 'No paid AI, no per-report cost' },
      { k: 7, u: 'stages', l: 'From dashcam frame to a verified repair' },
    ],
  }
}

export default function LandingPage() {
  const root = useRef(null)
  const heroImg = useRef(null)
  const reduced = useReducedMotion()
  const [data, setData] = useState(null)

  // pilot capture
  const [form, setForm] = useState({ name: '', email: '', city: '', message: '' })
  const [website, setWebsite] = useState('')      // honeypot
  const [sending, setSending] = useState(false)
  const [sent, setSent] = useState(false)
  const [err, setErr] = useState(null)

  useEffect(() => {
    let alive = true
    fetchLanding().then(d => { if (alive) setData(d) }).catch(() => {})
    return () => { alive = false }
  }, [])

  useReveal(root, !reduced)
  useParallax(heroImg, 0.18, !reduced)

  const setField = (k) => (e) => setForm(f => ({ ...f, [k]: e.target.value }))
  const submitPilot = async (e) => {
    e.preventDefault()
    setErr(null)
    if (!form.name.trim() || !form.email.trim() || !form.message.trim()) {
      setErr('Please fill in your name, email and a short note.')
      return
    }
    setSending(true)
    try {
      await requestPilot({ ...form, website })
      setSent(true)
    } catch (e2) {
      setErr(e2?.response?.data?.detail || 'Could not send just now. Please try again.')
    } finally {
      setSending(false)
    }
  }

  const { tiles, live } = statTiles(data)
  const recent = (data?.recent || []).filter(r => r && r.damage_type).slice(0, 6)
  const q = data?.quality

  return (
    <div className={`rdds-landing${reduced ? '' : ' anim'}`} ref={root}>
      <style>{CSS}</style>

      {/* ── hero ─────────────────────────────────────────────────────────── */}
      <header className="l-hero" id="top">
        <div className="l-hero-media" aria-hidden="true">
          <img ref={heroImg} src="/img/hero-a.jpg" width="1376" height="768"
               fetchpriority="high" decoding="async" alt="" />
          <span className="l-duotone" />
        </div>

        <div className="l-wrap l-hero-in">
          <div className="l-eyebrow" data-rv><span className="l-led" />Every car is a sensor</div>
          <h1 data-rv>A safer drive for<br /><em>the whole city.</em></h1>
          <p className="l-lead" data-rv>RDDS turns ordinary driving into a live map of road damage, so a city
            can fix the worst hazards first and everyone gets home safer. No new hardware, no cost per report.</p>
          <div className="l-cta" data-rv>
            <a className="btn btn-accent l-btn-lg" href="#pilot">Request a pilot</a>
            <a className="btn l-btn-lg" href="#method">See how it works</a>
          </div>

          <div className="l-board" data-rv>
            <div className="l-board-h">
              <span className="overline">{live ? 'Live board' : 'Product board'}</span>
              <span className="l-sweep" aria-hidden="true" />
            </div>
            <div className="l-board-grid">
              {tiles.map((t, i) => (
                <div className="l-kpi" key={i}>
                  <div className="l-kpi-k">
                    {t.plain ? t.k : <CountUp to={Number(t.k) || 0} reduced={reduced} />}
                    {t.u && <span className="l-kpi-u">{t.u}</span>}
                  </div>
                  <div className="l-kpi-l">{t.l}</div>
                </div>
              ))}
              {q && (
                <div className="l-kpi">
                  <div className="l-kpi-k" style={{ color: gradeColor(q.grade) }}>
                    {q.grade}<span className="l-kpi-u">{q.score}/100</span></div>
                  <div className="l-kpi-l">Road Quality Index, network-wide</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ── why it matters ───────────────────────────────────────────────── */}
      <section className="l-sec">
        <div className="l-wrap l-narrow" data-rv>
          <div className="overline l-over">Why it matters</div>
          <h2 className="l-big">A pothole is not a nuisance. It is a <em>hazard.</em></h2>
          <p className="l-body">Broken road surfaces cause blown tyres, sudden swerves, and the worst falls a
            cyclist can take. The damage a city cannot see is the damage that hurts someone. RDDS makes every
            ordinary drive add to a picture of where the road is failing, so the fix arrives before the harm does.</p>
        </div>
      </section>

      <section className="l-sec">
        <div className="l-wrap l-trio">
          {SAFETY.map((s, i) => (
            <article className="card l-panel" data-rv key={s.k} style={{ transitionDelay: `${i * 60}ms` }}>
              <h3>{s.k}</h3><p>{s.p}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ── method ───────────────────────────────────────────────────────── */}
      <section className="l-sec" id="method">
        <div className="l-wrap">
          <div className="l-head" data-rv>
            <h2>From a drive to a repair that holds.</h2>
            <p>Five stages, run in order. Each leaves a record the next one reads, so a street is only called
              safe once the road agrees.</p>
          </div>
          <ol className="l-chain">
            {LOOP.map((s, i) => (
              <li className="l-link" data-rv key={s.k} style={{ transitionDelay: `${i * 50}ms` }}>
                <span className="l-link-n mono">{s.n}</span>
                <div>
                  <h3>{s.k}</h3>
                  <p>{s.d}</p>
                  <code className="mono">{s.t}</code>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── severity, as something you drag ──────────────────────────────── */}
      <section className="l-sec" id="severity">
        <div className="l-wrap">
          <div className="l-head" data-rv>
            <h2>Severity you can argue with.</h2>
            <p>One honest scale, monitor to emergency. Drag the handle, throw it, or tap a stop.</p>
          </div>
          <div data-rv><SeverityDial reduced={reduced} /></div>

          {recent.length > 0 && (
            <div className="card l-feed" data-rv>
              <div className="l-feed-h"><span className="l-led" /><span className="overline">Most recent on record</span></div>
              <ul>
                {recent.map((r, i) => (
                  <li key={i}>
                    <span className="l-feed-s mono" style={{ '--c': SEVERITY[r.severity || 1]?.color }}>
                      S{r.severity || 1}</span>
                    <span className="l-feed-n">{label(r.damage_type)}</span>
                    {r.detection_count > 1 && <span className="l-feed-c mono">×{r.detection_count}</span>}
                    <span className="l-feed-t">{r.last_detected ? fmtDate(r.last_detected) : ''}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </section>

      {/* ── economics ────────────────────────────────────────────────────── */}
      <section className="l-sec" id="scale">
        <div className="l-wrap">
          <div className="l-head" data-rv>
            <h2>Built for a whole city, at no cost per report.</h2>
            <p>The economics are structural, not a promotion that expires.</p>
          </div>
          <div className="l-reasons">
            {REASONS.map((r, i) => (
              <article className="card l-panel" data-rv key={r.d} style={{ transitionDelay: `${i * 50}ms` }}>
                <span className="l-r-n mono">{r.d}</span><h3>{r.h}</h3><p>{r.p}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* ── pilot ────────────────────────────────────────────────────────── */}
      <section className="l-sec" id="pilot">
        <div className="l-wrap l-close">
          <div data-rv>
            <h2 className="l-big">Bring RDDS to <em>your city.</em></h2>
            <p className="l-body">Start with one vehicle and a week of footage. We hand back a map of your own
              roads, scored and ranked, before you commit to anything.</p>
            <ul className="l-close-list">
              <li>Worst hazards ranked in days, not months.</li>
              <li>Every model and map tile stays free and open.</li>
              <li>No lock-in: the data and the rules are yours to inspect.</li>
            </ul>
          </div>
          <div className="card l-form" data-rv style={{ transitionDelay: '60ms' }}>
            {sent ? (
              <div className="l-sent">
                <h3>Request received.</h3>
                <p>Thank you. We will be in touch about a pilot for {form.city || 'your city'} shortly.</p>
              </div>
            ) : (
              <form onSubmit={submitPilot} noValidate>
                <h3>Request a pilot</h3>
                {err && <div className="l-err" role="alert">{err}</div>}
                <label>Your name
                  <input className="input" value={form.name} onChange={setField('name')} autoComplete="name" required /></label>
                <label>Work email
                  <input className="input" type="email" value={form.email} onChange={setField('email')} autoComplete="email" required /></label>
                <label>City or road authority
                  <input className="input" value={form.city} onChange={setField('city')} autoComplete="organization" /></label>
                <label>What would you want to see first?
                  <textarea className="input" rows={3} value={form.message} onChange={setField('message')} required /></label>
                {/* honeypot: real people never see or fill this */}
                <input className="l-hp" tabIndex={-1} autoComplete="off" aria-hidden="true"
                       value={website} onChange={e => setWebsite(e.target.value)} />
                <button className="btn btn-accent l-btn-block" disabled={sending}>
                  {sending ? 'Sending…' : 'Request a pilot'}</button>
                <p className="l-fine">No obligation. We reply from a person, not a bot.</p>
              </form>
            )}
          </div>
        </div>
      </section>

      <footer className="l-foot">
        <div className="l-wrap l-foot-in">
          <span><b>RDDS</b> · Road Degradation Detection System · 2026</span>
          <span className="l-foot-links">
            <Link to="/pricing">Pricing</Link>
            <Link to="/developers">Developers</Link>
            <Link to="/login">Sign in</Link>
          </span>
          <span>A safer drive for the whole city.</span>
        </div>
      </footer>
    </div>
  )
}

const gradeColor = (g) => ({
  A: 'var(--s1)', B: 'var(--s2)', C: 'var(--s3)', D: 'var(--s4)', E: 'var(--s5)',
}[g] || 'var(--accent)')

/**
 * A figure that arrives on a critically damped spring. No overshoot: a KPI
 * that flies past its value and comes back reads as a glitch, however briefly.
 */
function CountUp({ to, reduced }) {
  const el = useRef(null)
  const [done, setDone] = useState(reduced)
  useEffect(() => {
    if (reduced || !el.current) return undefined
    const node = el.current
    let s = null
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting || s) return
      s = createSpring({
        damping: 1, response: 1.15, from: 0, precision: 0.5,
        onUpdate: (v) => { node.textContent = fmtNum(Math.round(v)) },
        onRest: () => setDone(true),
      })
      s.to(to)
      io.disconnect()
    }, { threshold: 0.3 })
    io.observe(node)
    return () => { io.disconnect(); s?.stop() }
  }, [to, reduced])
  return <span ref={el}>{done ? fmtNum(to) : '0'}</span>
}

/**
 * The severity scale as something you manipulate rather than read.
 *
 * Full gesture loop: pointer capture, 1:1 tracking from the grab offset,
 * rubber-band past the ends, momentum projection to choose the stop, then a
 * capped velocity handoff into the spring. Bounce is allowed here because a
 * flick preceded it; every other transition on this page is critically damped.
 */
function SeverityDial({ reduced }) {
  const rail = useRef(null)
  const thumb = useRef(null)
  const api = useRef(null)
  const [idx, setIdx] = useState(2)

  useEffect(() => {
    const rl = rail.current, th = thumb.current
    if (!rl || !th) return undefined

    let x = 0
    const paint = () => { th.style.transform = `translate3d(${x.toFixed(2)}px,-50%,0)` }
    const s = createSpring({ damping: 0.85, response: 0.36, precision: 0.3, onUpdate: v => { x = v; paint() } })

    const span = () => rl.getBoundingClientRect().width - th.offsetWidth
    const points = () => SEVERITY_BANDS.map((_, n) => (span() * n) / (SEVERITY_BANDS.length - 1))

    api.current = {
      go(n) {
        const p = points(), k = clamp(n, 0, p.length - 1)
        setIdx(k)
        if (reduced) s.set(p[k]); else s.to(p[k], { damping: 1, response: 0.4 })
      },
    }
    s.set(points()[2])

    const tv = trackVelocity()
    let active = null, grab = 0, origin = 0

    const onDown = (e) => {
      active = e.pointerId
      th.setPointerCapture(e.pointerId)
      s.hold()                       // grab it mid-glide and keep the speed
      grab = e.clientX
      origin = s.value               // respect where on the thumb they grabbed
      tv.clear(); tv.push(origin)
      th.classList.add('grabbing')
    }
    const onMove = (e) => {
      if (active !== e.pointerId) return
      const w = span()
      let raw = origin + (e.clientX - grab)
      if (raw < 0) raw = -rubberband(-raw, w)
      else if (raw > w) raw = w + rubberband(raw - w, w)
      s.set(raw)
      tv.push(raw)
      const p = points()
      setIdx(p.indexOf(nearest(clamp(raw, 0, w), p)))
    }
    const onUp = (e) => {
      if (active !== e.pointerId) return
      active = null
      th.classList.remove('grabbing')
      const v = tv.get(), w = span(), p = points()
      // Project the throw first, then snap to the stop nearest the landing
      // point — snapping from the release position throws the flick away.
      const landing = reduced ? s.value : s.value + project(v, 0.99)
      const target = nearest(clamp(landing, 0, w), p)
      setIdx(p.indexOf(target))
      if (reduced) s.set(target)
      else s.to(target, { velocity: handoff(v, target - s.value, 0.36, 2.2), damping: 0.8, response: 0.36 })
    }

    th.addEventListener('pointerdown', onDown)
    th.addEventListener('pointermove', onMove)
    th.addEventListener('pointerup', onUp)
    th.addEventListener('pointercancel', onUp)
    const onResize = () => s.set(points()[idx])
    window.addEventListener('resize', onResize)
    return () => {
      th.removeEventListener('pointerdown', onDown)
      th.removeEventListener('pointermove', onMove)
      th.removeEventListener('pointerup', onUp)
      th.removeEventListener('pointercancel', onUp)
      window.removeEventListener('resize', onResize)
      s.stop()
    }
    // Re-binding on every stop change would drop an in-flight gesture.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced])

  const band = SEVERITY_BANDS[idx]
  return (
    <div className="card l-dial" style={{ '--c': band.color }}>
      <div className="l-dial-rail" ref={rail}>
        <div className="l-dial-fill" style={{ width: `${(idx / (SEVERITY_BANDS.length - 1)) * 100}%` }} />
        {SEVERITY_BANDS.map((b, n) => (
          <button key={b.s} className={`l-dial-stop${n <= idx ? ' passed' : ''}`}
                  style={{ left: `${(n / (SEVERITY_BANDS.length - 1)) * 100}%`, '--sc': b.color }}
                  onPointerDown={() => api.current?.go(n)}
                  aria-label={`Severity ${b.s}, ${b.name}`}>
            <em className="mono">S{b.s}</em>
          </button>
        ))}
        <div className="l-dial-thumb mono" ref={thumb} role="slider" tabIndex={0}
             aria-valuemin={1} aria-valuemax={SEVERITY_BANDS.length} aria-valuenow={band.s}
             aria-valuetext={`S${band.s}, ${band.name}`} aria-label="Severity"
             onKeyDown={(e) => {
               if (e.key === 'ArrowRight') { e.preventDefault(); api.current?.go(idx + 1) }
               if (e.key === 'ArrowLeft') { e.preventDefault(); api.current?.go(idx - 1) }
             }}>
          S{band.s}
        </div>
      </div>
      <div className="l-dial-foot">
        <div className="l-dial-out" key={band.s}>
          <h3>{band.name}</h3>
          <p>{band.action}</p>
        </div>
        <p className="l-dial-note">Faded lane markings and worn crossings are capped low by class weight, not
          special-cased, so a cosmetic fault can never outrank a real danger. The band you land on is the one a
          crew is dispatched against.</p>
      </div>
    </div>
  )
}

/* ── scoped styles ────────────────────────────────────────────────────────
   Layout and type scale only. Every colour resolves to a design token, so the
   landing inherits any future theme change for free. */
const CSS = `
.rdds-landing{ --ease:cubic-bezier(.2,.7,.3,1); background:var(--bg); color:var(--text); }
.rdds-landing h1,.rdds-landing h2,.rdds-landing h3{
  font-family:var(--font-display);margin:0;font-weight:700;letter-spacing:-.032em;line-height:1.05}
.rdds-landing p{margin:0}
.rdds-landing a{color:inherit;text-decoration:none}
.rdds-landing em{font-style:normal;color:var(--accent)}
.rdds-landing .l-wrap{position:relative;max-width:1220px;margin:0 auto;padding:0 clamp(18px,4vw,48px)}
.rdds-landing .l-narrow{max-width:880px}
.rdds-landing .l-over{color:var(--accent);margin-bottom:14px}

.rdds-landing [data-rv]{opacity:1}
.rdds-landing.anim [data-rv]{opacity:0;transform:translateY(14px);
  transition:opacity .7s var(--ease),transform .7s var(--ease)}
.rdds-landing.anim [data-rv].in{opacity:1;transform:none}

.rdds-landing .l-led{width:7px;height:7px;border-radius:50%;background:var(--accent);flex:none}
.rdds-landing.anim .l-led{animation:l-led 2.6s var(--ease) infinite}
@keyframes l-led{0%{box-shadow:0 0 0 0 var(--accent-glow)}70%,100%{box-shadow:0 0 0 8px transparent}}

.rdds-landing .l-btn-lg{padding:13px 24px;font-size:14px;border-radius:var(--radius-lg)}
.rdds-landing .l-btn-block{width:100%;justify-content:center;margin-top:8px;padding:12px 20px}

/* hero */
.rdds-landing .l-hero{position:relative;padding:calc(var(--nav-h) + clamp(48px,8vw,96px)) 0 clamp(40px,6vw,72px);
  overflow:hidden}
.rdds-landing .l-hero-media{position:absolute;inset:0;z-index:0;overflow:hidden;background:var(--bg)}
.rdds-landing .l-hero-media img{width:100%;height:118%;object-fit:cover;object-position:50% 42%;
  filter:grayscale(.85) brightness(.62) contrast(1.2);opacity:.66;display:block}
.rdds-landing .l-duotone{position:absolute;inset:0;
  background:linear-gradient(180deg,rgba(14,13,11,.5),rgba(14,13,11,.8) 48%,var(--bg) 94%),
             radial-gradient(90% 70% at 18% 30%,rgba(192,71,42,.32),transparent 64%)}
.rdds-landing .l-hero-in{position:relative;z-index:1}
.rdds-landing .l-eyebrow{display:inline-flex;align-items:center;gap:9px;font-size:12.5px;font-weight:600;
  color:var(--text-dim);margin-bottom:20px}
.rdds-landing .l-hero h1{font-size:clamp(2.6rem,7vw,5.4rem);letter-spacing:-.045em;text-wrap:balance}
.rdds-landing .l-lead{margin-top:22px;max-width:52ch;font-size:16.5px;color:var(--text-dim);line-height:1.6}
.rdds-landing .l-cta{display:flex;gap:11px;flex-wrap:wrap;margin-top:28px}

/* board */
.rdds-landing .l-board{margin-top:clamp(38px,5vw,64px);border:1px solid var(--border);
  border-radius:var(--radius-xl);background:var(--bg-glass);overflow:hidden;
  backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  box-shadow:var(--edge-hi),var(--shadow-lg)}
.rdds-landing .l-board-h{position:relative;display:flex;align-items:center;padding:10px 18px;
  border-bottom:1px solid var(--border)}
.rdds-landing .l-sweep{position:absolute;left:0;bottom:-1px;width:120px;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent)}
.rdds-landing.anim .l-sweep{animation:l-sweep 5.5s var(--ease) infinite}
@keyframes l-sweep{0%{transform:translateX(-120px)}100%{transform:translateX(100vw)}}
.rdds-landing .l-board-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.rdds-landing .l-kpi{padding:22px 20px;border-right:1px solid var(--border)}
.rdds-landing .l-kpi:last-child{border-right:0}
.rdds-landing .l-kpi-k{font-family:var(--font-mono);font-size:clamp(1.6rem,3vw,2.4rem);font-weight:600;
  letter-spacing:-.04em;line-height:1;font-variant-numeric:tabular-nums;color:var(--text)}
/* Scoped to the unit suffix by class, NOT to any descendant span: CountUp
   renders its own span and a bare "span" selector shrinks the figure itself. */
.rdds-landing .l-kpi-k .l-kpi-u{display:inline-block;font-size:.38em;color:var(--text-muted);margin-left:6px;
  font-weight:400;letter-spacing:0}
.rdds-landing .l-kpi-l{font-size:12.5px;color:var(--text-dim);margin-top:11px;line-height:1.45;max-width:22ch}
@media(max-width:700px){.rdds-landing .l-kpi{border-right:0;border-bottom:1px solid var(--border)}}

/* sections */
.rdds-landing .l-sec{padding-top:clamp(60px,8vw,116px)}
.rdds-landing .l-big{font-size:clamp(1.9rem,4.4vw,3.2rem);letter-spacing:-.038em;max-width:20ch}
.rdds-landing .l-body{margin-top:20px;font-size:16px;color:var(--text-dim);max-width:56ch;line-height:1.65}
.rdds-landing .l-head{max-width:820px;margin-bottom:32px}
.rdds-landing .l-head h2{font-size:clamp(1.8rem,3.8vw,2.7rem);letter-spacing:-.036em}
.rdds-landing .l-head p{margin-top:12px;font-size:15.5px;color:var(--text-dim);max-width:54ch}

.rdds-landing .l-panel{padding:22px;transition:var(--transition)}
.rdds-landing .l-panel:hover{border-color:var(--border-bright);transform:translateY(-2px)}
.rdds-landing .l-panel h3{font-size:1.1rem;letter-spacing:-.024em}
.rdds-landing .l-panel p{margin-top:9px;font-size:14.5px;color:var(--text-dim);line-height:1.6}
.rdds-landing .l-r-n{display:block;font-size:11.5px;color:var(--accent);letter-spacing:.1em;margin-bottom:10px}
.rdds-landing .l-trio{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.rdds-landing .l-reasons{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
@media(max-width:860px){.rdds-landing .l-trio,.rdds-landing .l-reasons{grid-template-columns:1fr}}

/* method chain */
.rdds-landing .l-chain{list-style:none;margin:0;padding:0;border-top:1px solid var(--border)}
.rdds-landing .l-link{display:grid;grid-template-columns:64px 1fr;gap:clamp(14px,2.4vw,34px);padding:22px 0;
  border-bottom:1px solid var(--border);transition:background .26s var(--ease),padding-left .26s var(--ease)}
.rdds-landing .l-link:hover{background:var(--bg-card);padding-left:12px}
.rdds-landing .l-link-n{font-size:13px;font-weight:600;color:var(--accent);padding-top:4px}
.rdds-landing .l-link h3{font-size:clamp(1.2rem,2.2vw,1.6rem)}
.rdds-landing .l-link p{margin-top:8px;font-size:14.5px;color:var(--text-dim);max-width:62ch;line-height:1.6}
.rdds-landing .l-link code{display:inline-block;margin-top:11px;font-size:11.5px;color:var(--text-muted);
  background:var(--bg-card);padding:4px 10px;border-radius:var(--radius)}

/* the draggable severity bar */
.rdds-landing .l-dial{padding:clamp(24px,4vw,40px)}
.rdds-landing .l-dial-rail{position:relative;height:12px;border-radius:999px;background:var(--bg-card2);
  margin:30px 0 58px;touch-action:none}
.rdds-landing .l-dial-fill{position:absolute;left:0;top:0;bottom:0;border-radius:999px;background:var(--c);
  box-shadow:0 0 22px -4px var(--c);transition:width .38s var(--ease),background .38s var(--ease)}
.rdds-landing .l-dial-stop{position:absolute;top:50%;width:44px;height:44px;margin:-22px 0 0 -22px;border:0;
  background:none;padding:0;cursor:pointer;display:grid;place-items:center}
.rdds-landing .l-dial-stop::before{content:"";width:9px;height:9px;border-radius:50%;background:var(--bg);
  box-shadow:0 0 0 2px var(--border-bright);transition:box-shadow .3s var(--ease)}
.rdds-landing .l-dial-stop.passed::before{box-shadow:0 0 0 2px var(--sc)}
.rdds-landing .l-dial-stop em{position:absolute;top:28px;font-style:normal;font-size:11.5px;color:var(--text-muted)}
.rdds-landing .l-dial-thumb{position:absolute;left:0;top:50%;width:56px;height:56px;border-radius:50%;
  background:var(--c);color:var(--accent-contrast);display:grid;place-items:center;cursor:grab;touch-action:none;
  font-size:15px;font-weight:600;
  box-shadow:0 0 0 6px color-mix(in srgb,var(--c) 18%,transparent),0 18px 40px -12px var(--c);
  transition:background .38s var(--ease),box-shadow .22s var(--ease)}
.rdds-landing .l-dial-thumb.grabbing{cursor:grabbing;
  box-shadow:0 0 0 10px color-mix(in srgb,var(--c) 22%,transparent),0 22px 50px -12px var(--c)}
.rdds-landing .l-dial-foot{display:grid;grid-template-columns:1.05fr 1fr;gap:clamp(20px,4vw,52px);align-items:start}
.rdds-landing .l-dial-out{min-height:84px}
.rdds-landing .l-dial-out h3{font-size:1.55rem;color:var(--c);animation:l-swap .34s var(--ease) both}
.rdds-landing .l-dial-out p{margin-top:8px;font-size:16px;color:var(--text-dim);
  animation:l-swap .34s var(--ease) .04s both}
@keyframes l-swap{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
.rdds-landing .l-dial-note{padding-left:clamp(20px,4vw,52px);border-left:1px solid var(--border);
  font-size:13.5px;color:var(--text-muted);line-height:1.6}
@media(max-width:820px){
  .rdds-landing .l-dial-foot{grid-template-columns:1fr}
  .rdds-landing .l-dial-note{padding:16px 0 0;border-left:0;border-top:1px solid var(--border)}
}

/* recent feed */
.rdds-landing .l-feed{margin-top:26px;overflow:hidden;padding:0}
.rdds-landing .l-feed-h{display:flex;align-items:center;gap:9px;padding:11px 18px;border-bottom:1px solid var(--border)}
.rdds-landing .l-feed ul{list-style:none;margin:0;padding:0}
.rdds-landing .l-feed li{display:flex;align-items:center;gap:14px;padding:11px 18px;
  border-bottom:1px solid var(--border);font-size:14px}
.rdds-landing .l-feed li:last-child{border-bottom:0}
.rdds-landing .l-feed-s{font-size:11.5px;font-weight:600;color:var(--c);
  border:1px solid color-mix(in srgb,var(--c) 45%,transparent);padding:2px 8px;border-radius:var(--radius)}
.rdds-landing .l-feed-n{font-weight:600}
.rdds-landing .l-feed-c{font-size:12px;color:var(--text-muted)}
.rdds-landing .l-feed-t{margin-left:auto;font-size:12px;color:var(--text-muted)}

/* pilot */
.rdds-landing .l-close{display:grid;grid-template-columns:1.05fr .95fr;gap:clamp(28px,4vw,64px);align-items:start}
.rdds-landing .l-close-list{list-style:none;margin:26px 0 0;padding:0}
.rdds-landing .l-close-list li{padding:11px 0;border-top:1px solid var(--border);font-size:14.5px;color:var(--text-dim)}
.rdds-landing .l-form{padding:26px}
.rdds-landing .l-form h3{font-size:1.25rem;margin-bottom:18px}
.rdds-landing .l-form label{display:block;font-size:12.5px;font-weight:600;color:var(--text-dim);margin-bottom:13px}
.rdds-landing .l-form .input{width:100%;margin-top:6px;font-size:14px;padding:10px 12px}
.rdds-landing .l-form textarea.input{resize:vertical;font-family:var(--font-sans)}
.rdds-landing .l-hp{position:absolute!important;left:-9999px;width:1px;height:1px;opacity:0}
.rdds-landing .l-err{background:color-mix(in srgb,var(--red) 10%,transparent);
  border:1px solid color-mix(in srgb,var(--red) 35%,transparent);color:var(--red);
  border-radius:var(--radius);padding:9px 12px;font-size:13px;margin-bottom:13px}
.rdds-landing .l-fine{margin-top:12px;font-size:12px;color:var(--text-muted);text-align:center}
.rdds-landing .l-sent h3{font-size:1.4rem;color:var(--accent);margin-bottom:9px}
.rdds-landing .l-sent p{color:var(--text-dim);font-size:14.5px}
@media(max-width:860px){.rdds-landing .l-close{grid-template-columns:1fr}}

/* footer */
.rdds-landing .l-foot{margin-top:clamp(60px,8vw,110px);padding:0 0 48px;font-size:13px;color:var(--text-muted)}
.rdds-landing .l-foot-in{display:flex;flex-wrap:wrap;gap:12px 26px;justify-content:space-between;
  padding-top:24px;border-top:1px solid var(--border)}
.rdds-landing .l-foot b{color:var(--text)}
.rdds-landing .l-foot-links{display:flex;gap:18px}
.rdds-landing .l-foot-links a:hover{color:var(--accent)}

@media(prefers-reduced-motion:reduce){
  .rdds-landing .l-panel:hover{transform:none}
  .rdds-landing .l-link:hover{padding-left:0}
  .rdds-landing .l-dial-out h3,.rdds-landing .l-dial-out p{animation:none}
  .rdds-landing .l-sweep{animation:none}
}
`
