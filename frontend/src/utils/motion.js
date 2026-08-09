/**
 * frontend/src/utils/motion.js — the fluid-interface primitives.
 *
 * Dependency-free spring physics and gesture maths, in the parameterisation
 * Apple uses in "Designing Fluid Interfaces": damping ratio + response, not
 * mass/stiffness/damping. The whole point is that motion here can be *grabbed*
 * — it starts from the current on-screen value, inherits the pointer's
 * velocity, projects momentum forward, and can be reversed mid-flight.
 *
 *   createSpring    interruptible, velocity-carrying spring
 *   project         where a flick comes to rest (exponential decay)
 *   rubberband      progressive resistance past a boundary
 *   trackVelocity   short pointer history, so release velocity is real
 *   handoff         caps that velocity so a throw cannot overshoot its target
 *
 * Used by the landing page's severity bar and available to any page that wants
 * a draggable surface. CSS transitions remain the right tool for hover and
 * reveal; reach for a spring only when a finger is involved.
 */

import { useEffect, useState } from 'react'

/* ── spring presets ──────────────────────────────────────────────────────
   damping 1.0 = critically damped, no overshoot. Bounce is earned only when
   the gesture itself carried momentum; otherwise it reads as decoration. */
export const SPRING = {
  ui:     { damping: 1.0, response: 0.40 },  // move / reposition
  rotate: { damping: 0.8, response: 0.40 },  // rotation
  sheet:  { damping: 0.8, response: 0.30 },  // drawer / sheet
  snap:   { damping: 0.82, response: 0.36 }, // flick landing
}

export const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n))

/** Nearest value in `points` to `x` — the snap target after projection. */
export const nearest = (x, points) =>
  points.reduce((best, p) => (Math.abs(p - x) < Math.abs(best - x) ? p : best), points[0])

/**
 * A spring you can re-target at any moment. The new animation always starts
 * from the *presentation* value (where the pixel actually is) and keeps the
 * current velocity unless you replace it, which is what makes a reversal feel
 * like redirecting a moving object rather than hitting a wall.
 */
export function createSpring({
  damping = 1, response = 0.4, from = 0, precision = 0.01, onUpdate, onRest,
} = {}) {
  let x = from, v = 0, target = from
  let zeta = damping, omega = (2 * Math.PI) / response
  let raf = 0, last = 0, running = false

  const tick = (now) => {
    const dt = Math.min((now - last) / 1000, 1 / 30)   // a stalled tab must not explode the sim
    last = now
    const steps = Math.max(1, Math.ceil(dt / (1 / 240)))
    const h = dt / steps
    for (let i = 0; i < steps; i++) {
      const a = -omega * omega * (x - target) - 2 * zeta * omega * v
      v += a * h
      x += v * h
    }
    if (Math.abs(x - target) < precision && Math.abs(v) < precision * 12) {
      x = target; v = 0; running = false
      onUpdate?.(x, 0)
      onRest?.(x)
      return
    }
    onUpdate?.(x, v)
    raf = requestAnimationFrame(tick)
  }

  const start = () => {
    if (running) return
    running = true
    last = performance.now()
    raf = requestAnimationFrame(tick)
  }

  return {
    get value() { return x },
    get velocity() { return v },
    get target() { return target },
    /** Animate to `next` from wherever we are now. Pass velocity on release. */
    to(next, opts = {}) {
      target = next
      if (opts.velocity !== undefined) v = opts.velocity
      if (opts.damping !== undefined) zeta = opts.damping
      if (opts.response !== undefined) omega = (2 * Math.PI) / opts.response
      start()
    },
    /** Finger-driven 1:1 tracking — no physics while the pointer is down. */
    set(next) {
      running = false
      cancelAnimationFrame(raf)
      x = next; target = next
      onUpdate?.(x, v)
    },
    /** Grab a moving element: stop integrating but KEEP the velocity. */
    hold() { running = false; cancelAnimationFrame(raf) },
    stop() { running = false; cancelAnimationFrame(raf); v = 0 },
  }
}

/** Where a flick comes to rest. Exponential decay, not v²/2a. */
export function project(velocity, decelerationRate = 0.998) {
  return (velocity / 1000) * decelerationRate / (1 - decelerationRate)
}

/** Progressive resistance past an edge: it still moves, just less and less. */
export function rubberband(overshoot, dimension, constant = 0.55) {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot))
}

/** A real flick tops out around here; anything faster is a synthetic event. */
const MAX_FLICK = 4000   // px/s

/** Release velocity in px/s, measured over the last ~40ms of real movement. */
export function trackVelocity() {
  const hist = []
  return {
    push(pos, t = performance.now()) {
      hist.push([pos, t])
      if (hist.length > 6) hist.shift()
    },
    clear() { hist.length = 0 },
    get() {
      if (hist.length < 2) return 0
      const [p1, t1] = hist[hist.length - 1]
      let i = hist.length - 2
      while (i > 0 && t1 - hist[i][1] < 40) i--
      const [p0, t0] = hist[i]
      // A sub-4ms window divides by almost nothing and reports a five-figure
      // velocity — one stray sample would fling the element off the screen.
      const dt = Math.max(t1 - t0, 4) / 1000
      return clamp((p1 - p0) / dt, -MAX_FLICK, MAX_FLICK)
    },
  }
}

/**
 * Velocity handoff that cannot fling a spring past its target.
 *
 * A critically damped spring given more speed than its remaining distance can
 * absorb overshoots by roughly (v − ωd)/(ωe). Capping v at ~1.6·ωd keeps the
 * overshoot under a quarter of the distance, and the +150 floor stops short
 * hops arriving dead. The important case is a throw into a boundary: distance
 * collapses to zero, the cap collapses with it, and the element stops at the
 * edge instead of sailing through it.
 */
export function handoff(velocity, distance, response, slack = 1.6) {
  const omega = (2 * Math.PI) / response
  const cap = omega * Math.abs(distance) * slack + 150
  return clamp(velocity, -cap, cap)
}

/* ── environment ────────────────────────────────────────────────────────── */

/**
 * Only the accessibility signal. Unlike hooks/useMotionOk this does NOT turn
 * motion off on phones, because a gesture-driven surface has to keep tracking
 * the finger there. Decorative motion should still check useMotionOk.
 */
export function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => (typeof window !== 'undefined' && window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false))
  useEffect(() => {
    if (!window.matchMedia) return undefined
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const on = (e) => setReduced(e.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return reduced
}

/**
 * Scroll reveal for every [data-rv] under `ref`. Adds `.in` once, with a
 * safety timer so a missed observer can never leave a section blank.
 */
export function useReveal(ref, enabled = true) {
  useEffect(() => {
    const root = ref.current
    if (!root) return undefined
    const items = Array.from(root.querySelectorAll('[data-rv]'))
    if (!enabled || !('IntersectionObserver' in window)) {
      items.forEach(el => el.classList.add('in'))
      return undefined
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target) }
      })
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 })
    items.forEach(el => io.observe(el))
    const safety = setTimeout(() => items.forEach(el => el.classList.add('in')), 2000)
    return () => { io.disconnect(); clearTimeout(safety) }
  }, [ref, enabled])
}

/** Scroll parallax on a single element. rAF-throttled, transform only. */
export function useParallax(ref, factor = 0.22, enabled = true) {
  useEffect(() => {
    const el = ref.current
    if (!el || !enabled) return undefined
    let ticking = false
    const apply = () => {
      const y = window.scrollY
      if (y < window.innerHeight * 1.2) {
        el.style.transform = `translate3d(0, ${(y * factor).toFixed(1)}px, 0)`
      }
      ticking = false
    }
    const onScroll = () => { if (!ticking) { ticking = true; requestAnimationFrame(apply) } }
    window.addEventListener('scroll', onScroll, { passive: true })
    apply()
    return () => window.removeEventListener('scroll', onScroll)
  }, [ref, factor, enabled])
}
