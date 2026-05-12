/** 与后端 ensemble_exclude / 文档一致；供 Training 与 Ensemble 页共用 */

export const ENSEMBLE_BRANCH_IDS = ['RRTMIL', 'AMIL', 'WiKG', 'DSMIL', 'S4MIL']

/** 常用「用几路基线」子集（决策级 exclude） */
export const BRANCH_SUBSET_PRESETS = [
  {
    id: 'br-all',
    label: '五路全开',
    map: () => Object.fromEntries(ENSEMBLE_BRANCH_IDS.map((b) => [b, true])),
  },
  {
    id: 'br-rwd',
    label: '仅 RRT+WiKG+DSMIL',
    map: () => ({ RRTMIL: true, AMIL: false, WiKG: true, DSMIL: true, S4MIL: false }),
  },
  {
    id: 'br-mil3',
    label: '仅 AMIL+DSMIL+S4',
    map: () => ({ RRTMIL: false, AMIL: true, WiKG: false, DSMIL: true, S4MIL: true }),
  },
  {
    id: 'br-seq',
    label: '仅 WiKG+S4（图+序列）',
    map: () => ({ RRTMIL: false, AMIL: false, WiKG: true, DSMIL: false, S4MIL: true }),
  },
]
