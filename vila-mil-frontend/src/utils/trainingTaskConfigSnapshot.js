/** 与 Training.jsx 一致；由 ensembleExclude 反推参与融合的分支 */
export const ENSEMBLE_BRANCH_IDS = ['RRTMIL', 'AMIL', 'WiKG', 'DSMIL', 'S4MIL']

export const normalizeEnsembleExcludeKey = (name) => {
  const u = String(name).trim().toUpperCase().replace(/-/g, '_')
  if (u === 'S4') return 'S4MIL'
  if (u === 'WIKG') return 'WiKG'
  if (u === 'RRTMIL' || u === 'AMIL' || u === 'DSMIL' || u === 'S4MIL') return u
  return null
}

/** 未出现在 exclude 中的分支即参与（与训练页勾选「包含」一致） */
export const ensembleActiveBranches = (excluded) => {
  const ex = new Set((excluded || []).map(normalizeEnsembleExcludeKey).filter(Boolean))
  return ENSEMBLE_BRANCH_IDS.filter((b) => ex.has(b) === false)
}

export const formatFusionModeLabel = (fusion) => {
  const f = fusion ? String(fusion).toLowerCase() : ''
  if (f === 'gate') return '门控加权 (gate)'
  if (f === 'concat') return '对齐后拼接 (concat)'
  return fusion || '—'
}

export const formatDecisionFusionLabel = (v) => {
  const f = v ? String(v).toLowerCase() : ''
  if (f === 'avg_prob') return '简单概率均值 (avg_prob)'
  return v || '—'
}

const parseCmdArg = (cmd, flag) => {
  const esc = flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const m = String(cmd || '').match(new RegExp(`${esc}\\s+(\\S+)`, 'i'))
  return m ? m[1].trim() : ''
}

const readFinetuneEnsembleFlag = (task, cmd) => {
  const v = task?.finetuneEnsemble
  if (v === true || v === 1) return true
  if (v === false || v === 0) return false
  if (typeof v === 'string') {
    const s = v.trim().toLowerCase()
    if (['true', '1', 'yes', 'on'].includes(s)) return true
    if (['false', '0', 'no', 'off', ''].includes(s)) return false
  }
  if (/\b--finetune_ensemble\b/i.test(cmd)) return true
  return false
}

/**
 * 合并 tasks.json 字段与 command 字符串（与 Dashboard 任务详情一致）。
 * @param {Record<string, unknown>} task
 */
export const buildTaskConfigSnapshot = (task) => {
  if (!task) return null
  const cmd = String(task.command || '')
  const model = String(task.modelType || '')
  const early =
    typeof task.earlyStopping === 'boolean'
      ? task.earlyStopping
      : /\b--early_stopping\b/i.test(cmd)
  const fusionRaw = task.ensembleFusion || parseCmdArg(cmd, '--ensemble_fusion')
  const fusion = fusionRaw ? String(fusionRaw).toLowerCase() : ''
  const decisionRaw = task.decisionFusion || parseCmdArg(cmd, '--decision_fusion')
  const decisionFusion = decisionRaw ? String(decisionRaw).toLowerCase() : ''
  const dbwFromTask = task.decisionBranchWeights
  const decisionBranchWeights =
    dbwFromTask != null && String(dbwFromTask).trim() !== '' ? String(dbwFromTask).trim() : ''
  let exclude = []
  if (Array.isArray(task.ensembleExclude)) {
    exclude = task.ensembleExclude.map((s) => String(s).trim()).filter(Boolean)
  } else if (task.ensembleExclude && typeof task.ensembleExclude === 'string') {
    exclude = task.ensembleExclude.split(/[,;]/).map((s) => s.trim()).filter(Boolean)
  }
  if (exclude.length === 0 && /--ensemble_exclude\b/i.test(cmd)) {
    exclude = parseCmdArg(cmd, '--ensemble_exclude')
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
  }
  const finetune = readFinetuneEnsembleFlag(task, cmd)
  const wd = task.weightDecay != null && task.weightDecay !== '' ? task.weightDecay : parseCmdArg(cmd, '--reg')
  const k = task.kFolds != null ? task.kFolds : parseCmdArg(cmd, '--k')
  const me = task.maxEpochs != null ? task.maxEpochs : parseCmdArg(cmd, '--max_epochs')
  const lr = task.learningRate != null ? task.learningRate : parseCmdArg(cmd, '--lr')
  const seed = task.seed != null ? task.seed : parseCmdArg(cmd, '--seed')
  const mode = task.mode || parseCmdArg(cmd, '--mode')
  const ckptDir = (task.ensembleCkptDir || '').trim() || parseCmdArg(cmd, '--ensemble_ckpt_dir') || ''

  let ensembleExcludeFromCmd = null
  if (/--ensemble_exclude\b/i.test(cmd)) {
    const parts = parseCmdArg(cmd, '--ensemble_exclude')
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean)
    ensembleExcludeFromCmd = parts.length ? parts : []
  }

  return {
    cancer: task.cancer ?? '—',
    mode: mode || '—',
    model,
    maxEpochs: me ?? '—',
    learningRate: lr ?? '—',
    kFolds: k ?? '—',
    weightDecay: wd !== '' && wd != null ? wd : '—',
    seed: seed ?? '—',
    earlyStopping: early,
    batchSize: task.batchSize,
    repeatTotal: task.repeatTotal ?? task.repeat,
    ensembleFusion: fusion,
    decisionFusion,
    decisionBranchWeights,
    finetuneEnsemble: finetune,
    ensembleExclude: exclude,
    ensembleExcludeFromCmd,
    ensembleActiveBranches: ensembleActiveBranches(exclude),
    ensembleCkptDir: ckptDir,
  }
}
