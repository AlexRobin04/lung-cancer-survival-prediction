/** 与 Ensemble 页 / Prediction 页共享：蒸馏 tie-break 的「验证不足则 λ 回零」门控是否启用 */
export const ENSEMBLE_TIEBREAK_FALLBACK_LS_KEY = 'vilaMIL.ensembleTiebreakAllowFallback'

export function readEnsembleTiebreakAllowFallback() {
  try {
    const v = localStorage.getItem(ENSEMBLE_TIEBREAK_FALLBACK_LS_KEY)
    if (v === null) return true
    return v !== '0' && v !== 'false'
  } catch {
    return true
  }
}

export function writeEnsembleTiebreakAllowFallback(val) {
  try {
    localStorage.setItem(ENSEMBLE_TIEBREAK_FALLBACK_LS_KEY, val ? '1' : '0')
  } catch {
    // ignore
  }
}
