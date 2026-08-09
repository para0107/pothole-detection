/**
 * frontend/src/components/DamageGlyph.jsx — drawn icons for the damage classes.
 *
 * These replace a map of Unicode characters (〰 ═ ⬡ ⬤ ⛜ ◎ ▦ ∿) that stood in
 * for iconography. Borrowed glyphs are the wrong tool twice over: they render
 * differently on every platform's font stack, several of them fall back to a
 * tofu box, and none of them actually depicts road damage — a reader had to
 * learn an arbitrary symbol per class. Lucide has no road-damage set either,
 * so each class gets a purpose-drawn mark instead.
 *
 * Rules they all follow, so a row of them reads as one family:
 *   · 24×24 box, 1.6 stroke, round caps and joins, currentColor only
 *   · the road runs vertically (the direction of travel) in every glyph, so
 *     "longitudinal" and "transverse" are legible as directions, not labels
 *   · severity of the *fault* is never encoded here — colour carries that, and
 *     doubling it up would make the icon lie when the two disagree
 *   · worn markings fade along the stroke rather than adding a blur filter,
 *     which stays crisp at 14px and costs nothing to composite
 */

import React from 'react'

const GLYPHS = {
  // A crack running with the direction of travel.
  longitudinal_crack: (
    <path d="M12.6 3 L10.4 7.6 L13 10.6 L10.6 15.2 L13 18.2 L11.4 21" />
  ),

  // A crack running across the carriageway.
  transverse_crack: (
    <path d="M3 11.4 L7.6 13.4 L10.6 10.9 L15.2 13.1 L18.2 10.7 L21 12.4" />
  ),

  // Interlocking crazing — the alligator-hide pattern. Deliberately only four
  // strokes: the literal five-line net this started as collapsed into mush at
  // the 21px it actually ships at inside a row tile.
  alligator_crack: (
    <>
      <path d="M3.2 8.4 L10 10.4 L20.8 7.2" />
      <path d="M3.2 16.4 L10 14.4 L20.8 17.6" />
      <path d="M10 10.4 L10 14.4" />
      <path d="M16.2 8.6 L16.2 16.4" opacity="0.55" />
    </>
  ),

  // The old crack, faded, under a sealed band.
  repaired_crack: (
    <>
      <path d="M12.4 3 L10.4 7 L12.6 10" opacity="0.45" />
      <path d="M11.6 15 L12.8 18 L11.4 21" opacity="0.45" />
      <rect x="4.5" y="10" width="15" height="4.4" rx="2.2" />
    </>
  ),

  // An irregular depression with a broken rim.
  pothole: (
    <>
      <path d="M8.2 8.6 C10.4 6.6 15 7 16.4 9.4 C17.9 12 16.6 15.6 13.8 16.4
               C10.9 17.2 7.6 15.6 7.1 12.9 C6.8 11.2 7.2 9.5 8.2 8.6 Z" />
      <path d="M5.2 7.4 C6.4 5.6 8.4 4.6 10.6 4.4" opacity="0.45" />
      <path d="M18.6 16.8 C17.6 18.4 15.9 19.3 14 19.6" opacity="0.45" />
    </>
  ),

  // Zebra bars worn away from one side.
  pedestrian_crossing_blur: (
    <>
      <rect x="3.4" y="5.5" width="2.7" height="13" rx="1.3" />
      <rect x="8.4" y="5.5" width="2.7" height="13" rx="1.3" opacity="0.62" />
      <rect x="13.4" y="5.5" width="2.7" height="13" rx="1.3" opacity="0.38" />
      <rect x="18.4" y="5.5" width="2.7" height="13" rx="1.3" opacity="0.2" />
    </>
  ),

  // A centre line fading out along its length.
  lane_line_blur: (
    <>
      <path d="M12 3 L12 7.4" />
      <path d="M12 10.2 L12 14.2" opacity="0.55" />
      <path d="M12 17 L12 21" opacity="0.28" />
    </>
  ),

  // Concentric rings with seating notches.
  manhole_cover: (
    <>
      <circle cx="12" cy="12" r="8.2" />
      <circle cx="12" cy="12" r="3.3" />
      <path d="M12 3.8 L12 5.6" opacity="0.6" />
      <path d="M12 18.4 L12 20.2" opacity="0.6" />
    </>
  ),

  // A resurfaced square sitting proud of the surrounding surface.
  patchy_road: (
    <>
      <rect x="3.4" y="3.4" width="17.2" height="17.2" rx="2.4" opacity="0.35" />
      <path d="M7.6 8.2 L14.9 7.2 L17.4 11.4 L15.8 16.4 L8.6 16.9 Z" />
    </>
  ),

  // Two wheel troughs pressed into the surface.
  rutting: (
    <>
      <path d="M8.4 3.6 C6.9 8.6 6.6 15.2 7.8 20.4" />
      <path d="M15.6 3.6 C17.1 8.6 17.4 15.2 16.2 20.4" />
      <path d="M3.6 4.2 L3.6 19.8" opacity="0.35" />
      <path d="M20.4 4.2 L20.4 19.8" opacity="0.35" />
    </>
  ),
}

/** A generic mark for a class the schema gains before this file catches up. */
const FALLBACK = (
  <>
    <circle cx="12" cy="12" r="7.6" opacity="0.45" />
    <circle cx="12" cy="12" r="2.2" />
  </>
)

export default function DamageGlyph({ type, size = 20, title, ...rest }) {
  const glyph = GLYPHS[type] || FALLBACK
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      role={title ? 'img' : undefined}
      aria-hidden={title ? undefined : true}
      {...rest}
    >
      {title && <title>{title}</title>}
      {glyph}
    </svg>
  )
}

export const HAS_GLYPH = (type) => Object.prototype.hasOwnProperty.call(GLYPHS, type)
