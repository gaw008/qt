import { Navigate } from 'react-router-dom'

// Index page redirects to Dashboard
export default function Index() {
  return <Navigate to="/dashboard" replace />
}