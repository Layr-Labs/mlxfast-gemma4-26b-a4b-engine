// PagedAttentionResources.swift
//
// Catchable runtime-resource discovery for the paged-attention Metal source.
// SwiftPM's generated Bundle.module accessor calls fatalError when a release
// packager omits the target bundle. That is unacceptable in a provider
// process: resource eligibility must be decided while the backend is built,
// before registration or request admission.

import Foundation

private final class PagedAttentionBundleAnchor: NSObject {}

public enum PagedAttentionResourceError: Error, Equatable, CustomStringConvertible {
    case missing(resource: String, searchedRoots: [String])
    case ambiguous(resource: String, matches: [String])
    case unreadable(path: String)
    case invalid(path: String)

    public var description: String {
        switch self {
        case .missing(let resource, let roots):
            return "missing SwiftPM resource \(resource); searched \(roots.joined(separator: ", "))"
        case .ambiguous(let resource, let matches):
            return "ambiguous SwiftPM resource \(resource); matches \(matches.joined(separator: ", "))"
        case .unreadable(let path):
            return "unable to read paged-attention resource at \(path)"
        case .invalid(let path):
            return "invalid paged-attention Metal source at \(path)"
        }
    }
}

enum PagedAttentionResources {
    static let resourceName = "pagedattention.metal"

    /// Canonical sealed resource root when the current executable is inside
    /// a macOS app. No caller may add fallback roots in this context.
    ///
    /// The installer exposes the app executable through `bin/` symlinks
    /// (and operators add their own in /usr/local/bin), so the invocation
    /// path may not sit inside the .app at all. Resolve symlinks first —
    /// mirroring SelfUpdater's install-root resolution — otherwise a
    /// symlinked launch would silently miss the sealed resources.
    static func packagedAppResourcesURL(
        executableURL: URL? = Bundle.main.executableURL
    ) -> URL? {
        guard
            let executableURL = executableURL?
                .resolvingSymlinksInPath()
                .standardizedFileURL
        else {
            return nil
        }
        let macOS = executableURL.deletingLastPathComponent()
        let contents = macOS.deletingLastPathComponent()
        let app = contents.deletingLastPathComponent()
        guard macOS.lastPathComponent == "MacOS",
            contents.lastPathComponent == "Contents",
            app.pathExtension == "app"
        else { return nil }
        return app.appendingPathComponent(
            "Contents/Resources",
            isDirectory: true)
    }

    /// Development/test lookup roots. This path is used only when the
    /// executable is not packaged inside an app.
    static func developmentRoots(
        mainBundleURL: URL = Bundle.main.bundleURL,
        mainResourceURL: URL? = Bundle.main.resourceURL,
        executableURL: URL? = Bundle.main.executableURL
    ) -> [URL] {
        var roots = [mainBundleURL]
        if let mainResourceURL {
            roots.append(mainResourceURL)
        }
        if let executableURL {
            let executableDirectory = executableURL.deletingLastPathComponent()
            roots.append(executableDirectory)
            roots.append(
                executableDirectory.deletingLastPathComponent()
                    .appendingPathComponent("Resources", isDirectory: true))
        }
        let anchorBundle = Bundle(for: PagedAttentionBundleAnchor.self)
        let bundles = [anchorBundle] + Bundle.allBundles + Bundle.allFrameworks
        for bundle in bundles {
            roots.append(bundle.bundleURL)
            roots.append(bundle.bundleURL.deletingLastPathComponent())
            if let resourceURL = bundle.resourceURL {
                roots.append(resourceURL)
            }
            if let executableURL = bundle.executableURL {
                roots.append(executableURL.deletingLastPathComponent())
            }
        }
        if let argument = CommandLine.arguments.first, !argument.isEmpty {
            roots.append(
                URL(fileURLWithPath: argument).standardizedFileURL
                    .deletingLastPathComponent())
        }
        let workingDirectory = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath,
            isDirectory: true)
        roots.append(workingDirectory.appendingPathComponent(".build/debug", isDirectory: true))
        roots.append(workingDirectory.appendingPathComponent(".build/release", isDirectory: true))

        // Development/test fallback. Release artifacts resolve from the app
        // root above; this makes a standalone `swift test` find the bundle
        // beside its package build without using Bundle.module's fatal
        // accessor or baking an architecture-specific output path.
        var sourceAncestor = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
        for _ in 0 ..< 12 {
            if FileManager.default.fileExists(
                atPath: sourceAncestor.appendingPathComponent("Package.swift").path)
            {
                roots.append(
                    sourceAncestor.appendingPathComponent(
                        ".build/debug", isDirectory: true))
                roots.append(
                    sourceAncestor.appendingPathComponent(
                        ".build/release", isDirectory: true))
                break
            }
            sourceAncestor.deleteLastPathComponent()
        }

        // Dedupe on the RESOLVED path, in lockstep with `locate`. SwiftPM
        // writes `.build/debug` as a symlink to `.build/<triple>/debug`, so
        // the two roots appended above can name the same directory.
        var seen = Set<String>()
        return roots.filter {
            seen.insert($0.resolvingSymlinksInPath().standardizedFileURL.path).inserted
        }
    }

    /// Production-safe process lookup. A packaged executable is restricted
    /// to its sealed Contents/Resources tree; cwd and compile-time build
    /// roots are considered only for an unbundled development/test process.
    static func loadSourceForCurrentProcess(
        executableURL: URL? = Bundle.main.executableURL,
        developmentSearchRoots: [URL]? = nil,
        fileManager: FileManager = .default
    ) throws -> String {
        if let sealedRoot = packagedAppResourcesURL(executableURL: executableURL) {
            return try loadSource(
                roots: [sealedRoot],
                fileManager: fileManager)
        }
        return try loadSource(
            roots: developmentSearchRoots ?? developmentRoots(
                executableURL: executableURL),
            fileManager: fileManager)
    }

    static func locate(
        roots: [URL],
        fileManager: FileManager = .default
    ) throws -> URL {
        var matches: [URL] = []
        for root in roots {
            // Resolve before touching the filesystem. `.build/debug` is a
            // symlink to `.build/<triple>/debug`, and a URL-based directory
            // enumeration does NOT follow it: `contentsOfDirectory` returns
            // zero children, the resource inside the SwiftPM bundle is never
            // seen, preflight throws `.missing`, and the factory falls back
            // to contiguous WITHOUT a word. That silent fallback is the one
            // failure mode that makes a paged benchmark meaningless.
            // `packagedAppResourcesURL` above already resolves for the same
            // class of reason; this mirrors it.
            let resolved = root.resolvingSymlinksInPath()
            let direct = resolved.appendingPathComponent(resourceName)
            if fileManager.isReadableFile(atPath: direct.path) {
                matches.append(direct)
            }

            guard
                let children = try? fileManager.contentsOfDirectory(
                    at: resolved,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles])
            else { continue }
            for bundle in children where bundle.pathExtension == "bundle" {
                let candidate = bundle.appendingPathComponent(resourceName)
                if fileManager.isReadableFile(atPath: candidate.path) {
                    matches.append(candidate)
                }
            }
        }

        // Dedupe on CONTENT, not on path.
        //
        // Path dedupe cannot express what this check is actually for. The
        // `.ambiguous` error exists to stop the process loading an unknown
        // variant of the kernel source, so two matches only conflict when
        // their BYTES differ. Two are routinely reachable at once and are
        // identical: building `provider-swift` populates its own
        // `.build/<triple>/debug/mlx-swift-lm_MLXLMCommon.bundle`, and
        // building the submodule standalone populates
        // `libs/mlx-swift-lm/.build/...` — the source-ancestor walk above
        // finds the second. `pagedattention.metal` is copied verbatim as a
        // resource, never compiled, so both copies are byte-identical and
        // choosing either is the same choice.
        //
        // This also subsumes the symlink case: the same physical file reached
        // through `.build/debug` and `.build/<triple>/debug` reads identical
        // bytes and collapses here. Root order decides which URL is kept, so
        // the result is deterministic. A genuinely divergent resource still
        // throws.
        var seenContent = Set<Data>()
        var seenUnreadable = Set<String>()
        let unique = matches.filter { url in
            if let data = try? Data(contentsOf: url) {
                return seenContent.insert(data).inserted
            }
            // Unreadable: keep it distinct so the caller still gets a real
            // diagnostic rather than a silent drop.
            return seenUnreadable.insert(
                url.resolvingSymlinksInPath().standardizedFileURL.path
            ).inserted
        }
        guard !unique.isEmpty else {
            throw PagedAttentionResourceError.missing(
                resource: resourceName,
                searchedRoots: roots.map {
                    $0.resolvingSymlinksInPath().standardizedFileURL.path
                })
        }
        guard unique.count == 1 else {
            throw PagedAttentionResourceError.ambiguous(
                resource: resourceName,
                matches: unique.map(\.standardizedFileURL.path))
        }
        return unique[0]
    }

    static func loadSource(
        roots: [URL],
        fileManager: FileManager = .default
    ) throws -> String {
        let url = try locate(roots: roots, fileManager: fileManager)
        guard let source = try? String(contentsOf: url, encoding: .utf8) else {
            throw PagedAttentionResourceError.unreadable(path: url.path)
        }
        guard
            source.contains("namespace cbv2"),
            source.contains("paged_attention_part_impl"),
            source.contains("paged_kv_write_impl")
        else {
            throw PagedAttentionResourceError.invalid(path: url.path)
        }
        return source
    }
}
