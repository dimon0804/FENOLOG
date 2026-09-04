import { useEffect, useState } from 'react'

import { api } from '../api.js'

/**
 * Индикатор состояния внешних источников в шапке.
 *
 * Критерий требует не только продолжать работу при отказе источника, но и
 * сказать об этом. Поэтому здесь не «всё хорошо / всё плохо», а перечень с
 * последствием отказа каждого: пользователь должен понимать, что именно
 * перестало работать и можно ли продолжать.
 */
export default function SourcesHealth() {
  const [health, setHealth] = useState(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    let alive = true
    const load = () => api.health().then((h) => alive && setHealth(h)).catch(() => {})
    load()
    // Раз в минуту: ровно столько живёт кэш проверки на сервере, чаще спрашивать
    // бессмысленно — ответ будет тот же.
    const timer = setInterval(load, 60000)
    return () => { alive = false; clearInterval(timer) }
  }, [])

  if (!health) return <span className="small muted">проверяю источники…</span>

  const tone = { ok: 'ok', degraded: 'warn', down: 'bad' }[health.status] || 'warn'
  const label = {
    ok: 'Все источники доступны',
    degraded: 'Часть источников недоступна',
    down: 'Каталог снимков недоступен',
  }[health.status]

  return (
    <div style={{ position: 'relative' }}>
      <button className={`badge ${tone === 'ok' ? '' : tone}`} onClick={() => setOpen(!open)}>
        <span className={`dot ${tone}`} />
        {label}
      </button>

      {open && (
        <div
          className="card"
          style={{ position: 'absolute', right: 0, top: 36, width: 340, zIndex: 20, boxShadow: 'var(--shadow)' }}
        >
          {health.sources.map((source) => (
            <div key={source.key} style={{ marginBottom: 10 }}>
              <div className="row">
                <span className={`dot ${source.status === 'ok' ? 'ok' : source.required ? 'bad' : 'warn'}`} />
                <strong>{source.title}</strong>
              </div>
              <div className="small muted">{source.detail}</div>
              {source.consequence && <div className="small">{source.consequence}</div>}
            </div>
          ))}
          <div className="small muted">Проверка заняла {health.checked_in_seconds} с</div>
        </div>
      )}
    </div>
  )
}
