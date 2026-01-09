import { Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import { wsClient } from '@/lib/api'
import Layout from '@/components/Layout'
import ProtectedRoute from '@/components/ProtectedRoute'
import Dashboard from '@/pages/Dashboard'
import Markets from '@/pages/Markets'
import Trading from '@/pages/Trading'
import Risk from '@/pages/Risk'
import Login from '@/pages/Login'
import './App.css'

function App() {
  useEffect(() => {
    // Only initialize WebSocket if authenticated
    const token = localStorage.getItem('session_token')
    if (token) {
      wsClient.connect().catch(console.error)
    }

    return () => {
      wsClient.disconnect()
    }
  }, [])

  return (
    <Routes>
      {/* Public route - Login page */}
      <Route path="/login" element={<Login />} />

      {/* Protected routes - require authentication */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <Layout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/markets" element={<Markets />} />
                <Route path="/trading" element={<Trading />} />
                <Route path="/risk" element={<Risk />} />
              </Routes>
            </Layout>
          </ProtectedRoute>
        }
      />
    </Routes>
  )
}

export default App
