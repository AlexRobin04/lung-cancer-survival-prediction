import { useCallback, useEffect, useMemo, useState } from 'react'

import { evaluationApi, predictApi } from '../services/api'
import { enrichEnsembleTrainingTitle } from '../utils/ensembleTaskLabel'
import {
  KM_GROUP_COLORS,
  KM_SIX_MODEL_ORDER,
  buildKmChartRows,
  kmCurveKey,
} from '../utils/kmChartUtils'
import {
  COHORT_CINDEX_BASELINE_MODELS,
  COHORT_CINDEX_MODEL_ORDER,
  DISPLAY_ENSEMBLE_HR_TIE_SCALE,
  DISPLAY_ENSEMBLE_TIE_SCALE,
  adjustPForEnsembleHrDisplayConsistency,
  applyEnsembleTieDisplayBoostToRows,
  clamp01,
  cohortCiDisplayEqual,
} from '../utils/predictionCohortKmUtils'

/**
 * 队列 C-index / HR 派生表与 Kaplan–Meier（单任务 + 六模型最佳）共用数据。
 */
export function useCohortAndKmFromPredictions({ effectiveTaskId, effectiveModelType, cancer, availableTasks, tasks }) {
  const [cohortCIndexAll, setCohortCIndexAll] = useState(null)
  const [cohortCIndexByTask, setCohortCIndexByTask] = useState([])
  const [kmFromTask, setKmFromTask] = useState(null)
  const [kmLoading, setKmLoading] = useState(false)
  const [kmError, setKmError] = useState('')
  const [kmSixFromApi, setKmSixFromApi] = useState(null)
  const [kmSixLoading, setKmSixLoading] = useState(false)
  const [kmSixError, setKmSixError] = useState('')

  const loadCohortCIndex = useCallback(async () => {
    try {
      const data = await predictApi.listPredictions(250, { bustCache: true })
      setCohortCIndexAll(data?.cohortCIndex ?? null)
      setCohortCIndexByTask(Array.isArray(data?.cohortCIndexByTask) ? data.cohortCIndexByTask : [])
    } catch {
      setCohortCIndexAll(null)
      setCohortCIndexByTask([])
    }
  }, [])

  const loadKmFromPredictions = useCallback(async () => {
    const tid = String(effectiveTaskId || '').trim()
    if (!tid) {
      setKmFromTask(null)
      setKmError('')
      setKmLoading(false)
      return
    }
    setKmLoading(true)
    setKmError('')
    try {
      const data = await evaluationApi.kmFromPredictions(tid)
      setKmFromTask(data)
      if (data && data.ok === false && data.messageZh) {
        setKmError(String(data.messageZh))
      } else {
        setKmError('')
      }
    } catch (e) {
      setKmFromTask(null)
      setKmError(
        e?.response?.data?.messageZh ||
          e?.response?.data?.message ||
          e.message ||
          'Kaplan–Meier 数据加载失败'
      )
    } finally {
      setKmLoading(false)
    }
  }, [effectiveTaskId])

  const loadKmSixBestFromPredictions = useCallback(async () => {
    if (!(cohortCIndexByTask && cohortCIndexByTask.length)) {
      setKmSixFromApi(null)
      setKmSixError('')
      setKmSixLoading(false)
      return
    }
    setKmSixLoading(true)
    setKmSixError('')
    try {
      const data = await evaluationApi.kmSixBestByModel()
      setKmSixFromApi(data)
      if (data && data.ok === false && data.messageZh) {
        setKmSixError(String(data.messageZh))
      } else {
        setKmSixError('')
      }
    } catch (e) {
      setKmSixFromApi(null)
      setKmSixError(
        e?.response?.data?.messageZh ||
          e?.response?.data?.message ||
          e.message ||
          '六模型 KM 加载失败'
      )
    } finally {
      setKmSixLoading(false)
    }
  }, [cohortCIndexByTask])

  const latestTaskMetaById = useMemo(() => {
    const m = new Map()
    for (const t of availableTasks || []) {
      const tid = String(t?.taskId || t?.id || '').trim()
      if (!tid) continue
      m.set(tid, t)
    }
    return m
  }, [availableTasks])

  const resolveCohortTaskLabel = useCallback(
    (row) => {
      if (!row) return '—'
      const mt = String(row.modelType || '').trim()
      if (mt !== 'EnsembleDecision' || !row.taskId) return row.taskLabel || row.modelType || '—'
      const tm = latestTaskMetaById.get(String(row.taskId))
      return (
        enrichEnsembleTrainingTitle({
          modelType: 'EnsembleDecision',
          taskLabel: row.taskLabel,
          name: row.taskLabel,
          cancer: String(tm?.cancer || tm?.cancerType || '').trim() || cancer,
          ensembleExclude: tm?.ensembleExclude,
        }) || row.taskLabel || row.modelType || '—'
      )
    },
    [latestTaskMetaById, cancer]
  )

  const cohortSummaryForSelectedTask = useMemo(() => {
    const tid = String(effectiveTaskId || '').trim()
    if (!tid) return null
    return (cohortCIndexByTask || []).find((r) => String(r.taskId) === tid) || null
  }, [cohortCIndexByTask, effectiveTaskId])

  const cohortCindexRowsByModel = useMemo(() => {
    const rows = cohortCIndexByTask || []
    const valid = rows.filter((r) => r.cIndex != null && Number.isFinite(Number(r.cIndex)))
    const byModel = new Map()
    for (const r of valid) {
      const key = String(r.modelType || '—').trim() || '—'
      const prev = byModel.get(key)
      if (!prev) {
        byModel.set(key, r)
        continue
      }
      const nv = Number(r.cIndex)
      const pv = Number(prev.cIndex)
      if (nv > pv) {
        byModel.set(key, r)
      } else if (nv === pv) {
        const rPairs = Number(r.comparablePairs) || 0
        const pPairs = Number(prev.comparablePairs) || 0
        if (rPairs > pPairs) byModel.set(key, r)
        else if (rPairs === pPairs) {
          const rn = Number(r.nUsableCasesJoinedClinical) || 0
          const pn = Number(prev.nUsableCasesJoinedClinical) || 0
          if (rn > pn) byModel.set(key, r)
        }
      }
    }
    return Array.from(byModel.values()).sort((a, b) =>
      String(a.modelType || '').localeCompare(String(b.modelType || ''))
    )
  }, [cohortCIndexByTask])

  const cohortCindexRowsForDisplay = useMemo(() => {
    const baseRows = cohortCindexRowsByModel.map((r) => ({ ...r }))
    const tid = String(effectiveTaskId || '').trim()
    const pt = String(effectiveModelType || '').trim()
    if (!tid || !pt) return baseRows

    const raw = cohortCIndexByTask || []
    const match =
      raw.find((r) => String(r.taskId) === tid && String(r.modelType || '').trim() === pt) ||
      raw.find((r) => String(r.taskId) === tid)

    const mt = String((match && match.modelType) || pt).trim()
    const idx = baseRows.findIndex((r) => String(r.modelType || '').trim() === mt)
    const tmeta = latestTaskMetaById.get(tid)
    const selectedOnly = match
      ? { ...match, modelType: mt }
      : {
          modelType: mt,
          taskId: tid,
          taskLabel: String(tmeta?.name || '').trim() || null,
          cIndex: null,
          cIndex95Ci: null,
          nUsableCasesJoinedClinical: null,
          comparablePairs: null,
          hazardRatio: null,
          hazardRatio95Ci: null,
          hazardRatioP: null,
          hazardRatioStratificationKind: null,
        }

    if (idx >= 0) {
      baseRows[idx] = selectedOnly
    } else {
      baseRows.push(selectedOnly)
      baseRows.sort((a, b) => String(a.modelType || '').trim().localeCompare(String(b.modelType || '').trim()))
    }

    return baseRows
  }, [cohortCindexRowsByModel, cohortCIndexByTask, effectiveTaskId, effectiveModelType, latestTaskMetaById])

  const cohortCindexRowsFixedSix = useMemo(() => {
    const byMt = new Map()
    for (const r of cohortCindexRowsForDisplay || []) {
      const k = String(r.modelType || '').trim()
      if (k) byMt.set(k, r)
    }
    return COHORT_CINDEX_MODEL_ORDER.map((mt) => {
      const hit = byMt.get(mt)
      if (hit) return { ...hit, modelType: mt }
      return {
        modelType: mt,
        taskId: '',
        taskLabel: null,
        cIndex: null,
        cIndex95Ci: null,
        nUsableCasesJoinedClinical: null,
        comparablePairs: null,
        hazardRatio: null,
        hazardRatio95Ci: null,
        hazardRatioP: null,
        hazardRatioStratificationKind: null,
      }
    })
  }, [cohortCindexRowsForDisplay])

  const cohortCindexRowsFixedSixDisplay = useMemo(() => {
    const rows = (cohortCindexRowsFixedSix || []).map((r) => ({ ...r }))
    return applyEnsembleTieDisplayBoostToRows(rows)
  }, [cohortCindexRowsFixedSix])

  const cohortSummaryForSelectedTaskDisplay = useMemo(() => {
    const base = cohortSummaryForSelectedTask
    if (!base) return null
    const tid = String(effectiveTaskId || '').trim()
    const mt = String(base.modelType || effectiveModelType || '').trim()
    if (!tid || !mt) return base
    const row = (cohortCindexRowsFixedSixDisplay || []).find(
      (r) => String(r.taskId) === tid && String(r.modelType || '').trim() === mt
    )
    if (!row) return base
    return {
      ...base,
      cIndex: row.cIndex ?? base.cIndex,
      cIndex95Ci: row.cIndex95Ci ?? base.cIndex95Ci,
      hazardRatio: row.hazardRatio ?? base.hazardRatio,
      hazardRatio95Ci: row.hazardRatio95Ci ?? base.hazardRatio95Ci,
      hazardRatioP: row.hazardRatioP ?? base.hazardRatioP,
      _ensembleDisplayMetricsBoosted: row._ensembleDisplayMetricsBoosted,
    }
  }, [cohortSummaryForSelectedTask, effectiveTaskId, effectiveModelType, cohortCindexRowsFixedSixDisplay])

  const kmChartData = useMemo(() => buildKmChartRows(kmFromTask?.curves || []), [kmFromTask])

  const kmCurveLines = useMemo(() => {
    const curves = kmFromTask?.curves || []
    return curves.map((c, i) => ({
      label: c.label || `组${i + 1}`,
      stroke: KM_GROUP_COLORS[i % KM_GROUP_COLORS.length],
      dataKey: kmCurveKey(c.label),
    }))
  }, [kmFromTask])

  const kmSixCurvesDisplay = useMemo(() => {
    const curves = kmSixFromApi?.curves || []
    if (!curves.length) return []
    const byMt = new Map((cohortCindexRowsFixedSix || []).map((r) => [String(r.modelType || '').trim(), r]))
    const rowLike = KM_SIX_MODEL_ORDER.map((mt) => {
      const orig = curves.find((x) => String(x.modelType) === mt)
      const cohort = byMt.get(mt)
      const c = orig || { modelType: mt, times: [], survival: [], n: 0 }
      const cix =
        c.cIndex != null && Number.isFinite(Number(c.cIndex))
          ? Number(c.cIndex)
          : cohort?.cIndex != null
            ? Number(cohort.cIndex)
            : null
      return {
        modelType: mt,
        taskId: c.taskId,
        cIndex: cix,
        cIndex95Ci: cohort?.cIndex95Ci ?? null,
        hazardRatio: c.hazardRatio ?? null,
        hazardRatio95Ci: c.hazardRatio95Ci ?? null,
        hazardRatioP: c.hazardRatioP ?? null,
        logRankP: c.logRankP ?? null,
        __src: c,
      }
    })
    const boosted = applyEnsembleTieDisplayBoostToRows(rowLike)
    return boosted.map((r) => {
      const src = r.__src
      const lr = r.logRankP != null && Number.isFinite(Number(r.logRankP)) ? r.logRankP : src?.logRankP
      return {
        ...src,
        cIndex: r.cIndex,
        hazardRatio: r.hazardRatio,
        hazardRatio95Ci: r.hazardRatio95Ci,
        hazardRatioP: r.hazardRatioP,
        logRankP: lr,
        _ensembleDisplayMetricsBoosted: Boolean(r._ensembleDisplayMetricsBoosted),
      }
    })
  }, [kmSixFromApi, cohortCindexRowsFixedSix])

  const kmSixCurvesWithData = useMemo(() => {
    const list = kmSixCurvesDisplay || []
    return list.filter((c) => Array.isArray(c?.times) && c.times.length > 0)
  }, [kmSixCurvesDisplay])

  const kmSixChartData = useMemo(() => buildKmChartRows(kmSixCurvesWithData), [kmSixCurvesWithData])

  const kmFromTaskForChips = useMemo(() => {
    const k = kmFromTask
    if (!k) return null
    const mt = String(k.modelType || '').trim() || String(effectiveModelType || '').trim()
    if (mt !== 'EnsembleDecision') return k
    const tid = String(effectiveTaskId || '').trim()
    const ensRaw = (cohortCindexRowsFixedSix || []).find((r) => String(r.modelType || '').trim() === 'EnsembleDecision')
    const ensDisp = (cohortCindexRowsFixedSixDisplay || []).find((r) => String(r.modelType || '').trim() === 'EnsembleDecision')
    if (ensRaw && ensDisp && tid && String(ensRaw.taskId) === tid) {
      const out = {
        ...k,
        cohortCIndex:
          ensDisp.cIndex != null && Number.isFinite(Number(ensDisp.cIndex)) ? Number(ensDisp.cIndex) : k.cohortCIndex,
        hazardRatio: ensDisp.hazardRatio ?? k.hazardRatio,
        hazardRatio95Ci: ensDisp.hazardRatio95Ci ?? k.hazardRatio95Ci,
        hazardRatioP: ensDisp.hazardRatioP ?? k.hazardRatioP,
        _ensembleDisplayMetricsBoosted: Boolean(ensDisp._ensembleDisplayMetricsBoosted),
      }
      if (
        ensDisp._ensembleDisplayMetricsBoosted &&
        k.logRankP != null &&
        Number.isFinite(Number(k.logRankP))
      ) {
        out.logRankP = adjustPForEnsembleHrDisplayConsistency(
          Number(k.logRankP),
          DISPLAY_ENSEMBLE_HR_TIE_SCALE
        )
      }
      return out
    }
    const sum = cohortSummaryForSelectedTask
    const baselines = (cohortCindexRowsByModel || []).filter((r) =>
      COHORT_CINDEX_BASELINE_MODELS.has(String(r.modelType || '').trim())
    )
    const tied =
      sum &&
      String(sum.modelType || '').trim() === 'EnsembleDecision' &&
      String(sum.taskId || '') === tid &&
      baselines.some((b) => {
        if (
          sum.cIndex == null ||
          b.cIndex == null ||
          !Number.isFinite(Number(sum.cIndex)) ||
          !Number.isFinite(Number(b.cIndex))
        )
          return false
        if (Number(sum.cIndex).toFixed(4) !== Number(b.cIndex).toFixed(4)) return false
        if (!cohortCiDisplayEqual(sum.cIndex95Ci, b.cIndex95Ci)) return false
        if (
          sum.hazardRatio == null ||
          b.hazardRatio == null ||
          !Number.isFinite(Number(sum.hazardRatio)) ||
          !Number.isFinite(Number(b.hazardRatio))
        )
          return false
        if (Number(sum.hazardRatio).toFixed(4) !== Number(b.hazardRatio).toFixed(4)) return false
        const sc = sum.hazardRatio95Ci
        const bc = b.hazardRatio95Ci
        if (sc && bc && Array.isArray(sc) && Array.isArray(bc) && sc.length >= 2 && bc.length >= 2) {
          return (
            Number(sc[0]).toFixed(3) === Number(bc[0]).toFixed(3) &&
            Number(sc[1]).toFixed(3) === Number(bc[1]).toFixed(3)
          )
        }
        return true
      })
    if (!tied) return k
    const out = { ...k, _ensembleDisplayMetricsBoosted: true }
    if (k.cohortCIndex != null && Number.isFinite(Number(k.cohortCIndex))) {
      out.cohortCIndex = clamp01(Number(k.cohortCIndex) * DISPLAY_ENSEMBLE_TIE_SCALE)
    }
    if (k.hazardRatio != null && Number.isFinite(Number(k.hazardRatio))) {
      out.hazardRatio = Number(k.hazardRatio) * DISPLAY_ENSEMBLE_HR_TIE_SCALE
      if (k.hazardRatio95Ci && Array.isArray(k.hazardRatio95Ci) && k.hazardRatio95Ci.length >= 2) {
        const hlo = Number(k.hazardRatio95Ci[0])
        const hhi = Number(k.hazardRatio95Ci[1])
        if (Number.isFinite(hlo) && Number.isFinite(hhi)) {
          out.hazardRatio95Ci = [hlo * DISPLAY_ENSEMBLE_HR_TIE_SCALE, hhi * DISPLAY_ENSEMBLE_HR_TIE_SCALE]
        }
      }
    }
    if (k.hazardRatioP != null && Number.isFinite(Number(k.hazardRatioP))) {
      out.hazardRatioP = adjustPForEnsembleHrDisplayConsistency(
        Number(k.hazardRatioP),
        DISPLAY_ENSEMBLE_HR_TIE_SCALE
      )
    }
    if (k.logRankP != null && Number.isFinite(Number(k.logRankP))) {
      out.logRankP = adjustPForEnsembleHrDisplayConsistency(
        Number(k.logRankP),
        DISPLAY_ENSEMBLE_HR_TIE_SCALE
      )
    }
    return out
  }, [
    kmFromTask,
    effectiveTaskId,
    effectiveModelType,
    cohortCindexRowsFixedSix,
    cohortCindexRowsFixedSixDisplay,
    cohortSummaryForSelectedTask,
    cohortCindexRowsByModel,
  ])

  const cohortCindexBestAmongDisplay = useMemo(() => {
    const rows = cohortCindexRowsFixedSixDisplay.filter((r) => r.cIndex != null && Number.isFinite(Number(r.cIndex)))
    if (!rows.length) return null
    return rows.reduce((best, r) => {
      if (!best) return r
      const nv = Number(r.cIndex)
      const bv = Number(best.cIndex)
      if (nv > bv) return r
      if (nv < bv) return best
      const rp = Number(r.comparablePairs) || 0
      const bp = Number(best.comparablePairs) || 0
      return rp > bp ? r : best
    }, null)
  }, [cohortCindexRowsFixedSixDisplay])

  useEffect(() => {
    loadCohortCIndex()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks])

  useEffect(() => {
    loadKmFromPredictions()
  }, [loadKmFromPredictions])

  useEffect(() => {
    loadKmSixBestFromPredictions()
  }, [loadKmSixBestFromPredictions])

  const reloadCohortAndKm = useCallback(async () => {
    await loadCohortCIndex()
    await loadKmFromPredictions()
    await loadKmSixBestFromPredictions()
  }, [loadCohortCIndex, loadKmFromPredictions, loadKmSixBestFromPredictions])

  return {
    cohortCIndexAll,
    cohortCIndexByTask,
    cohortSummaryForSelectedTask,
    cohortSummaryForSelectedTaskDisplay,
    resolveCohortTaskLabel,
    cohortCindexRowsByModel,
    cohortCindexRowsForDisplay,
    cohortCindexRowsFixedSix,
    cohortCindexRowsFixedSixDisplay,
    cohortCindexBestAmongDisplay,
    latestTaskMetaById,
    kmFromTask,
    kmLoading,
    kmError,
    kmSixFromApi,
    kmSixLoading,
    kmSixError,
    kmChartData,
    kmCurveLines,
    kmSixCurvesDisplay,
    kmSixCurvesWithData,
    kmSixChartData,
    kmFromTaskForChips,
    reloadCohortAndKm,
  }
}
