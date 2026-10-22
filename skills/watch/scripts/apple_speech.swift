import AVFoundation
import Foundation
import Speech

let url = URL(fileURLWithPath: CommandLine.arguments[1])
let wanted = CommandLine.arguments.count > 2 ? Locale(identifier: CommandLine.arguments[2]) : Locale.current
guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: wanted) else {
    FileHandle.standardError.write("Apple Speech does not support locale \(wanted.identifier)\n".data(using: .utf8)!)
    exit(2)
}
let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [.audioTimeRange])
if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
    try await request.downloadAndInstall()
}
let analyzer = SpeechAnalyzer(modules: [transcriber])
let collector = Task {
    var segments: [[String: Any]] = []
    for try await result in transcriber.results {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespaces)
        if text.isEmpty { continue }
        segments.append(["start": result.range.start.seconds, "end": result.range.end.seconds, "text": text])
    }
    return segments
}
if let last = try await analyzer.analyzeSequence(from: try AVAudioFile(forReading: url)) {
    try await analyzer.finalizeAndFinish(through: last)
} else {
    await analyzer.cancelAndFinishNow()
}
let segments = try await collector.value
print(String(data: try JSONSerialization.data(withJSONObject: ["segments": segments, "language": locale.identifier]), encoding: .utf8)!)
