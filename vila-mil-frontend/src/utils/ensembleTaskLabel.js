/** 与后端 api_server._ENSEMBLE_BRANCH_ABBREV_ORDER 一致：纳入支路首字母，五路全纳入 → AWRDS */
export const ENSEMBLE_ABBREV_BRANCH_ORDER = ['AMIL', 'WiKG', 'RRTMIL', 'DSMIL', 'S4MIL']

export function ensembleIncludedAbbrevFromTask(task) {
  if (String(task?.modelType || '').trim() !== 'EnsembleDecision') return ''
  const ex = new Set((task?.ensembleExclude || []).map((x) => String(x).trim()).filter(Boolean))
  const s = ENSEMBLE_ABBREV_BRANCH_ORDER.filter((b) => !ex.has(b))
    .map((b) => b.charAt(0).toUpperCase())
    .join('')
  return s || 'X'
}

/** Recent Training Tasks 下拉：EnsembleDecision-AWRDS-completed-时间 */
export function formatTrainingHistorySelectLabel(task) {
  const st = String(task?.status || '').trim()
  const time = task?.startedAt || task?.queuedAt || ''
  const best = task?.isBestForModel ? '-Best' : ''
  if (String(task?.modelType || '').trim() === 'EnsembleDecision') {
    const abbr = ensembleIncludedAbbrevFromTask(task)
    return `EnsembleDecision-${abbr}-${st}${best}-${time}`
  }
  const mt = String(task?.modelType || '').trim()
  return `${mt}-${st}${best}-${time}`
}

/** 管理历史弹窗：含癌种 */
export function formatTrainingHistoryDialogLabel(task) {
  const c = String(task?.cancer || '').trim()
  const st = String(task?.status || '').trim()
  const time = task?.startedAt || task?.queuedAt || ''
  const best = task?.isBestForModel ? '-Best' : ''
  if (String(task?.modelType || '').trim() === 'EnsembleDecision') {
    const abbr = ensembleIncludedAbbrevFromTask(task)
    return `EnsembleDecision-${abbr}-${c}-${st}${best}-${time}`
  }
  const mt = String(task?.modelType || '').trim()
  return `${mt}-${c}-${st}${best}-${time}`
}

/** 模型评估 /evaluation/runs 下拉：EnsembleDecision-AWRDS — LUSC — completed */
export function formatEvalRunMenuLabel(run) {
  const c = String(run?.cancer || '').trim()
  const st = String(run?.status || '').trim()
  if (String(run?.modelType || '').trim() === 'EnsembleDecision') {
    const abbr = ensembleIncludedAbbrevFromTask(run)
    return `EnsembleDecision-${abbr} — ${c} — ${st}`
  }
  const mt = String(run?.modelType || '').trim()
  return `${mt} — ${c} — ${st}`
}

/** Prediction 手动选 Task：EnsembleDecision-AWRDS — LUSC — epochs:n — ckpt:k */
export function formatPredictionTaskMenuLabel(task) {
  const c = String(task?.cancer || '').trim()
  const ep = task?.maxEpochs ?? '—'
  const ck = task?.checkpointCount ?? 0
  if (String(task?.modelType || '').trim() === 'EnsembleDecision') {
    const abbr = ensembleIncludedAbbrevFromTask(task)
    return `EnsembleDecision-${abbr} — ${c} — epochs:${ep} — ckpt:${ck}`
  }
  const mt = String(task?.modelType || '').trim()
  return `${mt} — ${c} — epochs:${ep} — ckpt:${ck}`
}

/**
 * Prediction「最佳」选 Model：仅当 o 上带有 ensembleExclude 时缩写才准确；
 * 否则 o 往往只有 { key, cancer, modelType }，会误判为五路全开 → AWRDS。
 */
export function formatPredictionModelPickLabel(o) {
  const c = String(o?.cancer || '').trim()
  if (String(o?.modelType || '').trim() === 'EnsembleDecision') {
    const abbr = ensembleIncludedAbbrevFromTask({
      modelType: 'EnsembleDecision',
      ensembleExclude: o?.ensembleExclude ?? [],
    })
    return `EnsembleDecision-${abbr} — ${c}`
  }
  return `${o?.modelType} — ${c}`
}

/** 与 ModelEvaluation.pickBestCompletedRun 一致 */
export function pickBestCompletedEvalRun(list, cancer, modelType) {
  const c = String(cancer || '').trim()
  const m = String(modelType || '').trim()
  const cands = (list || []).filter(
    (r) =>
      String(r?.status || '').toLowerCase() === 'completed' &&
      String(r?.cancer || '').trim() === c &&
      String(r?.modelType || '').trim() === m
  )
  if (cands.length === 0) return null
  const byLoss = [...cands].sort((a, b) => {
    const la = Number(a?.loss)
    const lb = Number(b?.loss)
    const va = Number.isFinite(la) ? la : Number.POSITIVE_INFINITY
    const vb = Number.isFinite(lb) ? lb : Number.POSITIVE_INFINITY
    if (va !== vb) return va - vb
    return String(b?.taskId || '').localeCompare(String(a?.taskId || ''))
  })
  return byLoss[0] || cands[0]
}

/** 与 ModelEvaluation.pickBestRunForDisplay 一致：先 completed 最优 loss，否则同癌种+模型按 startedAtTs 最近 */
export function pickRepresentativeEvalRun(completedList, allRuns, cancer, modelType) {
  const fromDone = pickBestCompletedEvalRun(completedList, cancer, modelType)
  if (fromDone) return fromDone
  const c = String(cancer || '').trim()
  const m = String(modelType || '').trim()
  const cands = (allRuns || []).filter(
    (r) => String(r?.cancer || '').trim() === c && String(r?.modelType || '').trim() === m
  )
  if (cands.length === 0) return null
  const ts = (r) => Number(r?.startedAtTs) || 0
  return [...cands].sort((a, b) => ts(b) - ts(a))[0] || null
}

/** 与 Prediction.pickBestAvailableTask 一致（用于「最佳」Model 行展示缩写） */
export function pickRepresentativePredictionTaskForAbbrev(tasks, cancer, modelType) {
  const c = String(cancer || '').trim()
  const m = String(modelType || '').trim()
  const cands = (tasks || []).filter(
    (t) =>
      String(t?.cancer || t?.cancerType || '').trim() === c &&
      String(t?.modelType || t?.model_type || '').trim() === m
  )
  if (cands.length === 0) return null
  const marked = cands.find((t) => Boolean(t?.isBestForModel))
  if (marked) return marked
  const byLoss = [...cands].sort((a, b) => {
    const la = Number(a?.loss)
    const lb = Number(b?.loss)
    const va = Number.isFinite(la) ? la : Number.POSITIVE_INFINITY
    const vb = Number.isFinite(lb) ? lb : Number.POSITIVE_INFINITY
    if (va !== vb) return va - vb
    return String(b?.startedAt || '').localeCompare(String(a?.startedAt || ''))
  })
  return byLoss[0] || cands[0]
}

/** 评估页「最佳」模型下拉：用 runs 上的 ensembleExclude，与侧栏 formatEnsembleDecisionHeader 一致 */
export function formatBestModelPickLabelEval(o, { runs, completedRuns, taskId, bestModelKey }) {
  const c = String(o?.cancer || '').trim()
  const mt = String(o?.modelType || '').trim()
  if (mt !== 'EnsembleDecision') return `${mt} — ${c}`
  const wantKey = `${c}__${mt}`
  const keyMatches = String(bestModelKey || '') === wantKey
  if (keyMatches && taskId) {
    const cur = (runs || []).find((r) => String(r.taskId) === String(taskId))
    if (cur && String(cur.cancer || '').trim() === c && String(cur.modelType || '').trim() === 'EnsembleDecision') {
      return `EnsembleDecision-${ensembleIncludedAbbrevFromTask(cur)} — ${c}`
    }
  }
  const done = completedRuns ?? (runs || []).filter((r) => String(r?.status || '').toLowerCase() === 'completed')
  const pick = pickRepresentativeEvalRun(done, runs, c, 'EnsembleDecision')
  if (pick) return `EnsembleDecision-${ensembleIncludedAbbrevFromTask(pick)} — ${c}`
  return formatPredictionModelPickLabel(o)
}

/** 预测页「最佳」Model 下拉：用任务列表上的 ensembleExclude */
export function formatBestModelPickLabelPrediction(o, { availableTasks, taskId, pickedModelKey }) {
  const c = String(o?.cancer || '').trim()
  const mt = String(o?.modelType || '').trim()
  if (mt !== 'EnsembleDecision') return `${mt} — ${c}`
  const wantKey = `${c}__${mt}`
  const keyMatches = String(pickedModelKey || '') === wantKey
  if (keyMatches && taskId) {
    const cur = (availableTasks || []).find((t) => String(t.taskId) === String(taskId))
    if (
      cur &&
      String(cur.cancer || cur.cancerType || '').trim() === c &&
      String(cur.modelType || cur.model_type || '').trim() === 'EnsembleDecision'
    ) {
      return `EnsembleDecision-${ensembleIncludedAbbrevFromTask(cur)} — ${c}`
    }
  }
  const pick = pickRepresentativePredictionTaskForAbbrev(availableTasks, c, 'EnsembleDecision')
  if (pick) return `EnsembleDecision-${ensembleIncludedAbbrevFromTask(pick)} — ${c}`
  return formatPredictionModelPickLabel(o)
}

/** 模型评估侧栏小标题：EnsembleDecision-AWRDS */
export function formatEnsembleDecisionHeader(snapshotOrTask) {
  const mt = String(snapshotOrTask?.model || snapshotOrTask?.modelType || '').trim()
  if (mt !== 'EnsembleDecision') return 'EnsembleDecision'
  const abbr = ensembleIncludedAbbrevFromTask({
    modelType: 'EnsembleDecision',
    ensembleExclude: snapshotOrTask?.ensembleExclude,
  })
  return `EnsembleDecision-${abbr}`
}

/**
 * 队列 C-index 表副标题：把旧名「… EnsembleDecision Training」补成带支路缩写（需传入 ensembleExclude）。
 */
export function enrichEnsembleTrainingTitle(partial) {
  const mt = String(partial?.modelType || partial?.model || '').trim()
  if (mt !== 'EnsembleDecision') return String(partial?.name || partial?.taskLabel || '').trim() || null
  const name = String(partial?.name || partial?.taskLabel || '').trim()
  const cancer = String(partial?.cancer || '').trim()
  const abbr = ensembleIncludedAbbrevFromTask({
    modelType: 'EnsembleDecision',
    ensembleExclude: partial?.ensembleExclude,
  })
  if (name.includes(`EnsembleDecision-${abbr}`)) return name
  const patched = name.replace(/\bEnsembleDecision Training\b/, `EnsembleDecision-${abbr} Training`)
  if (patched !== name) return patched
  if (cancer && name === `${cancer} EnsembleDecision Training`) return `${cancer} EnsembleDecision-${abbr} Training`
  if (name) return name
  return cancer ? `${cancer} EnsembleDecision-${abbr} Training` : `EnsembleDecision-${abbr} Training`
}
