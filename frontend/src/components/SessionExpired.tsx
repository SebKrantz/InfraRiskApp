import { useEffect, useState } from 'react'
import { Button } from './ui/button'
import { SESSION_EXPIRED_EVENT } from '../lib/session'

export default function SessionExpired() {
  const [expired, setExpired] = useState(false)

  useEffect(() => {
    const show = () => setExpired(true)
    window.addEventListener(SESSION_EXPIRED_EVENT, show)
    return () => window.removeEventListener(SESSION_EXPIRED_EVENT, show)
  }, [])

  if (!expired) return null
  return (
    <div className="fixed inset-x-0 top-4 z-[100] flex justify-center px-4">
      <div role="alert" className="flex items-center gap-4 rounded-md bg-gray-900 px-4 py-3 text-sm text-white shadow-lg">
        <span>Your sign-in has expired.</span>
        <Button size="sm" onClick={() => window.location.reload()}>
          Reload
        </Button>
      </div>
    </div>
  )
}
