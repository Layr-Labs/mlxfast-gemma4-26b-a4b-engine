// Entry point for the `BenchCBv2` executable.
//
// Deliberately the whole target: the driver, its option parser and its report
// builders live in the BenchCBv2Core LIBRARY, and Tests/BenchCBv2Tests imports
// that library. A test target that depends on an *executable* target makes
// SwiftPM run that binary as the swift-testing host and pass it
// `--test-bundle-path`; `BenchOptions.parse` rejects the unknown flag and
// exits, which aborts the swift-testing pass for the entire package — 728
// `@Test` cases, including the paged-KV CI gates, silently executed nothing.
// Keep this file a shim so nothing can ever depend on the executable again.

import BenchCBv2Core

@main
struct BenchCBv2RealModel {
    static func main() async {
        await BenchCBv2Driver.run()
    }
}
