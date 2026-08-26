import Foundation
import PackagePlugin

/// Generates `BenchBuildRevision.swift` for BenchCBv2 before every build, so
/// the benchmark report can name the revision the executable was *built* from.
///
/// A prebuild command (not a build command) because there is no input file to
/// key on: the revision changes when the checkout moves, not when a source
/// file does, so it must be recomputed on every build invocation. The script
/// rewrites the generated file only when the value actually changes, so a
/// no-op build stays a no-op.
@main
struct BenchRevisionStamp: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        let outputDirectory = context.pluginWorkDirectoryURL
            .appending(path: "GeneratedSources")
        try FileManager.default.createDirectory(
            at: outputDirectory, withIntermediateDirectories: true)
        let script = context.package.directoryURL
            .appending(path: "scripts/stamp-bench-revision.sh")
        return [
            .prebuildCommand(
                displayName: "Stamp BenchCBv2 build revision",
                executable: URL(fileURLWithPath: "/bin/sh"),
                arguments: [
                    script.path(percentEncoded: false),
                    context.package.directoryURL.path(percentEncoded: false),
                    outputDirectory.appending(path: "BenchBuildRevision.swift")
                        .path(percentEncoded: false),
                ],
                outputFilesDirectory: outputDirectory)
        ]
    }
}
