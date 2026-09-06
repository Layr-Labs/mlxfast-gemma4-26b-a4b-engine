import Foundation

/// MAGIC-HALF (MH4): the integer-code dequantization step of the three MMA8
/// decode planes, expressed through the `half` binade `[1024, 2048)` instead of
/// an integer-to-float conversion.
///
/// `0x6400` is `1024.0h` with a zero mantissa. Every half in `[1024, 2048)` has
/// a unit ULP, so `as_type<half>(ushort(0x6400u | code))` is exactly the half
/// `1024 + code` for every code `0 ... 1023`, and subtracting the exactly
/// representable `1024.0h` returns `code` with no rounding at any step. The
/// construction breaks at 1025; a 4-bit code is `0 ... 15` and an 8-bit code is
/// `0 ... 255`, both far inside the range.
///
/// The three planes it is applied to are the ones whose dequantized value feeds
/// a `simdgroup_matrix` fragment and therefore stays live across the whole
/// accumulate:
///
/// - `AttentionOQMVV1` `MMA8_STEP` (4-bit)
/// - `AttentionQKVMMA8V1` `MMA8_STEP` (4-bit)
/// - `DenseMLPQMVV1` `MMA8_STEP8` (8-bit)
///
/// It is deliberately NOT applied to the tied LM head or to the sliding q4 KV
/// walk. Both consume the dequantized value immediately in float, so the
/// shorter live range buys nothing there and the two extra half-domain
/// instructions only lengthen the issue stream.
///
/// Kill switch: `DARKBLOOM_GEMMA4_MMA8_HALF_DEQUANT=0` restores
/// `float(code)` in all three headers, byte for byte.
enum Gemma4MMA8HalfDequant {
    static let on: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_MMA8_HALF_DEQUANT"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    /// The `MMA8_DQ(V)` macro pair, emitted into each MMA8 kernel header.
    static let macro: String = """
        #define MMA8_HALF_DEQ \(on ? 1 : 0)
        #if MMA8_HALF_DEQ
        #define MMA8_DQ(V) float(as_type<half>(ushort(0x6400u | (V))) - 1024.0h)
        #else
        #define MMA8_DQ(V) float(V)
        #endif
        """
}
