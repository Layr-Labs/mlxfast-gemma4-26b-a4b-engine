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

        var seen = Set<String>()
        return roots.filter {
            seen.insert($0.resolvingSymlinksInPath().standardizedFileURL.path).inserted
        }
    }

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

        var seenContent = Set<Data>()
        var seenUnreadable = Set<String>()
        let unique = matches.filter { url in
            if let data = try? Data(contentsOf: url) {
                return seenContent.insert(data).inserted
            }
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
