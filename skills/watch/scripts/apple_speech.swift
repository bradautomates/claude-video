// On-device transcription with macOS 26 SpeechAnalyzer.
// Usage: apple_speech <audio-file> [language-or-locale] [title/description sample]
// Without a language, it is detected from the text sample, else the system locale is used.
// Prints {"language": "<bcp47>", "language_source": "given|detected|system", "segments": [...]}.
import AVFoundation
import Foundation
import NaturalLanguage
import Speech

func fail(_ message: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

let args = CommandLine.arguments
guard args.count >= 2 else { fail("usage: apple_speech <audio-file> [language-or-locale] [text-sample]", 2) }
let audioURL = URL(fileURLWithPath: args[1])
let given = args.count > 2 ? args[2].trimmingCharacters(in: .whitespaces) : ""
let sample = args.count > 3 ? args[3] : ""
let system = Locale.current

// A bare language ("pt") that matches the system language keeps the system region (pt-BR, not pt-PT).
func supported(_ hint: String) async -> Locale? {
    let requested = Locale(identifier: hint)
    let sameLanguage = requested.language.languageCode == system.language.languageCode
    return await SpeechTranscriber.supportedLocale(equivalentTo: requested.region == nil && sameLanguage ? system : requested)
}

var resolved: Locale?
var source = "system"
if !given.isEmpty {
    guard let locale = await supported(given) else { fail("Apple Speech does not support \(given)", 2) }
    resolved = locale
    source = "given"
} else if !sample.isEmpty {
    let recognizer = NLLanguageRecognizer()
    recognizer.processString(sample)
    if let (language, probability) = recognizer.languageHypotheses(withMaximum: 1).first, probability >= 0.5,
       let locale = await supported(language.rawValue) {
        resolved = locale
        source = "detected"
    }
}
if resolved == nil { resolved = await SpeechTranscriber.supportedLocale(equivalentTo: system) }
guard let locale = resolved else { fail("Apple Speech does not support the system locale \(system.identifier(.bcp47))", 2) }

let transcriber = SpeechTranscriber(
    locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [.audioTimeRange])
do {
    let installed = await SpeechTranscriber.installedLocales.contains { $0.identifier(.bcp47) == locale.identifier(.bcp47) }
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        if !installed {
            FileHandle.standardError.write("downloading the \(locale.identifier(.bcp47)) speech model…\n".data(using: .utf8)!)
        }
        try await request.downloadAndInstall()
    }
} catch {
    fail("Apple Speech model install failed: \(error.localizedDescription)")
}

// Finalized results can span a minute of audio. Split them on sentence ends, long pauses,
// or a length cap, using each word's own time range.
let maxSegmentSeconds = 20.0
let pauseSeconds = 1.5
let sentenceEnds: Set<Character> = [".", "!", "?", "…", "。", "！", "？"]

let collector = Task { () -> [[String: Any]] in
    var segments: [[String: Any]] = []
    var text = ""
    var start: Double?
    var end = 0.0
    func flush() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // A wrong-language model emits bare punctuation for speech it cannot read; drop it.
        if let s = start, trimmed.contains(where: { $0.isLetter || $0.isNumber }) {
            segments.append(["start": s, "end": max(end, s), "text": trimmed])
        }
        text = ""
        start = nil
    }
    for try await result in transcriber.results {
        for run in result.text.runs {
            let piece = String(result.text[run.range].characters)
            if let range = run.audioTimeRange {
                let s = range.start.seconds, e = range.end.seconds
                if let open = start, s - end > pauseSeconds || e - open > maxSegmentSeconds { flush() }
                if start == nil { start = s }
                end = max(end, e)
            }
            text += piece
            if let last = piece.trimmingCharacters(in: .whitespaces).last, sentenceEnds.contains(last) { flush() }
        }
        flush()
    }
    return segments
}

do {
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let file = try AVAudioFile(forReading: audioURL)
    if let last = try await analyzer.analyzeSequence(from: file) {
        try await analyzer.finalizeAndFinish(through: last)
    } else {
        await analyzer.cancelAndFinishNow()
    }
    let segments = try await collector.value
    let payload: [String: Any] = ["language": locale.identifier(.bcp47), "language_source": source, "segments": segments]
    print(String(data: try JSONSerialization.data(withJSONObject: payload), encoding: .utf8)!)
} catch {
    fail("Apple Speech transcription failed: \(error.localizedDescription)")
}
