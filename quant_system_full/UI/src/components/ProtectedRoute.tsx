import { Navigate, useLocation } from 'react-router-dom'
import { ReactNode, useEffect, useState } from 'react'

interface ProtectedRouteProps {
  children: ReactNode
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const [isValidating, setIsValidating] = useState(true)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const validateSession = async () => {
      const token = localStorage.getItem('session_token')

      if (!token) {
        setIsAuthenticated(false)
        setIsValidating(false)
        return
      }

      try {
        const apiBaseUrl = import.meta.env.DEV ? '' : (import.meta.env.VITE_API_BASE_URL || '')
        const response = await fetch(`${apiBaseUrl}/api/auth/verify`, {
          headers: {
            'Authorization': `Session ${token}`
          }
        })

        const data = await response.json()
        setIsAuthenticated(data.valid === true)
      } catch (error) {
        // If verification fails, clear invalid session and redirect to login
        console.error('Session verification failed:', error)
        localStorage.removeItem('session_token')
        setIsAuthenticated(false)
      } finally {
        setIsValidating(false)
      }
    }

    validateSession()
  }, [])

  if (isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-white">Validating session...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    // Redirect to login page, preserving the intended destination
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}
