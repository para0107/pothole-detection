import React, { Suspense, lazy } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Navbar from './components/Navbar'
import { AuthProvider, useAuth } from './context/AuthContext'
import { Spinner, CenterState } from './components/ui'
import CityGate from './components/CityGate'
import OnboardingTour from './components/OnboardingTour'

// Every page is a lazy chunk: the shell (navbar + auth) loads instantly and a
// user only downloads the code for pages they actually open. Vite splits the
// heavy vendors (leaflet, recharts, animation libs) into their own chunks too,
// and the assistant's model runtimes are dynamic imports on top of that.
const HomePage       = lazy(() => import('./pages/HomePage'))
// MapPage and QualityPage are no longer routed directly — MapWorkspacePage
// mounts one or the other as the Detections / Quality layer of a single map.
const MapWorkspacePage = lazy(() => import('./pages/MapWorkspacePage'))
const StatsPage      = lazy(() => import('./pages/StatsPage'))
const DamagePage     = lazy(() => import('./pages/DamagePage'))
const IngestionPage  = lazy(() => import('./pages/IngestionPage'))
const AboutPage      = lazy(() => import('./pages/AboutPage'))
const LivePage       = lazy(() => import('./pages/LivePage'))
const LoginPage      = lazy(() => import('./pages/LoginPage'))
const RegisterPage   = lazy(() => import('./pages/RegisterPage'))
const AdminPage      = lazy(() => import('./pages/AdminPage'))
const ImpactPage     = lazy(() => import('./pages/ImpactPage'))
// Triage and WorkOrders are no longer routed directly — OperationsPage mounts
// them as its two tabs and is the single destination for the repair workflow.
const OperationsPage = lazy(() => import('./pages/OperationsPage'))
const AssistantPage  = lazy(() => import('./pages/AssistantPage'))
const PricingPage    = lazy(() => import('./pages/PricingPage'))
const DevelopersPage = lazy(() => import('./pages/DevelopersPage'))
const LandingPage    = lazy(() => import('./pages/LandingPage'))

/**
 * Everything except the public pages requires a session. Accounts without a
 * city (Google first login, legacy rows) must pick one before using the app:
 * the maps and municipality scoping depend on it.
 */
function RequireAuth({ children }) {
  const { isAuthed, booting, user } = useAuth()
  const location = useLocation()
  if (booting) {
    return (
      <div style={{ paddingTop: 'var(--nav-h)', height: '100%' }}>
        <CenterState><Spinner label="Restoring session…" /></CenterState>
      </div>
    )
  }
  if (!isAuthed) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }
  if (!user?.city) {
    return (
      <>
        {children}
        <CityGate />
      </>
    )
  }
  return (
    <>
      {children}
      <OnboardingTour />
    </>
  )
}

/**
 * Survey and operations pages (Map, Explorer, Stats, Repairs, Upload, Triage,
 * Work orders, Quality) are for municipality operators and admins only;
 * citizens are sent back to Command. The backend enforces the same rule on the
 * underlying endpoints.
 */
function RequireOperator({ children }) {
  const { isOperator } = useAuth()
  if (!isOperator) return <Navigate to="/" replace />
  return children
}

function RouteFallback() {
  return (
    <div style={{ paddingTop: 'var(--nav-h)', height: '100%' }}>
      <CenterState><Spinner label="Loading…" /></CenterState>
    </div>
  )
}

const operatorRoute = (element) => (
  <RequireAuth><RequireOperator>{element}</RequireOperator></RequireAuth>
)

/**
 * The front door at "/". Logged-out visitors get the public marketing landing;
 * signed-in users get their command dashboard (behind the usual city gate and
 * onboarding tour that RequireAuth adds).
 */
function Root() {
  const { isAuthed, booting } = useAuth()
  if (booting) return <RouteFallback />
  if (!isAuthed) return <LandingPage />
  return <RequireAuth><HomePage /></RequireAuth>
}

export default function App() {
  return (
    <AuthProvider>
      <Navbar />
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          {/* Public */}
          <Route path="/login"      element={<LoginPage />} />
          <Route path="/register"   element={<RegisterPage />} />
          <Route path="/pricing"    element={<PricingPage />} />
          <Route path="/developers" element={<DevelopersPage />} />

          {/* Public front door: marketing landing when logged out, dashboard when signed in */}
          <Route path="/"          element={<Root />} />

          <Route path="/live"      element={<RequireAuth><LivePage /></RequireAuth>} />
          <Route path="/impact"    element={<RequireAuth><ImpactPage /></RequireAuth>} />
          <Route path="/assistant" element={<RequireAuth><AssistantPage /></RequireAuth>} />
          <Route path="/about"     element={<RequireAuth><AboutPage /></RequireAuth>} />
          <Route path="/admin"     element={<RequireAuth><AdminPage /></RequireAuth>} />

          {/* Operator only */}
          <Route path="/map"        element={operatorRoute(<MapWorkspacePage />)} />
          <Route path="/stats"      element={operatorRoute(<StatsPage />)} />
          <Route path="/damage"     element={operatorRoute(<DamagePage />)} />
          <Route path="/operations" element={operatorRoute(<OperationsPage />)} />
          <Route path="/ingest"     element={operatorRoute(<IngestionPage />)} />

          {/* Retired routes. Explorer/Priority became the two views of Damage
              and Triage/WorkOrders the two tabs of Operations; these keep old
              links, bookmarks and anything a crew wrote down still working. */}
          <Route path="/explorer"   element={<Navigate to="/damage?view=table" replace />} />
          <Route path="/priority"   element={<Navigate to="/damage" replace />} />
          <Route path="/triage"     element={<Navigate to="/operations" replace />} />
          <Route path="/workorders" element={<Navigate to="/operations?tab=orders" replace />} />
          <Route path="/quality"    element={<Navigate to="/map?layer=quality" replace />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AuthProvider>
  )
}
