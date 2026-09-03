// MOE-MMA8-001 REMOVED (ranked-fidelity autopsy, submission 09616f38).
//
// The matrix-unit gather tier changed batch-8 token values vs the incumbent
// QMV tiers (88/192 token diffs on identical inputs; ranked pairs need
// exact tape equality), so batch-8 MoE gather stays on the incumbent tiers.
// This file remains only to keep the target file set stable; it defines no
// symbols and affects no dispatch. The SwitchLayers admission hunks were
// reverted alongside.
enum Gemma4MoEGatherMMA8Removed {
    static let note = "see above"
}
