import { useEffect, useState } from 'react'
import { dataApi } from '../services/api'
import { CANCER_OPTIONS } from '../constants/trainingOptions'

export default function useCancerOptions(defaultCancer = 'LUSC') {
  const [cancerOptions, setCancerOptions] = useState(CANCER_OPTIONS)
  const [cancer, setCancer] = useState(defaultCancer)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const ds = await dataApi.getDatasets()
        const raw = ds?.cancers || ds?.datasets || []
        const vals = Array.isArray(raw)
          ? raw
              .map((x) => (typeof x === 'string' ? x : x?.cancer || x?.id || x?.value || ''))
              .map((s) => String(s || '').trim())
              .filter(Boolean)
          : []
        if (!cancelled && vals.length > 0) {
          const uniq = [...new Set(vals)]
          const opts = uniq.map((v) => ({ value: v, label: v }))
          setCancerOptions(opts)
          setCancer((prev) => (uniq.includes(prev) ? prev : uniq[0]))
          return
        }
      } catch {
        // fallback below
      }
      if (!cancelled) {
        setCancerOptions(CANCER_OPTIONS)
        setCancer((prev) => (CANCER_OPTIONS.some((o) => o.value === prev) ? prev : CANCER_OPTIONS[0]?.value || defaultCancer))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [defaultCancer])

  return { cancerOptions, cancer, setCancer }
}
