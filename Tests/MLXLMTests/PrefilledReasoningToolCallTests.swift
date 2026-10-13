// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

struct PrefilledReasoningToolCallTests {
    private let tools: [[String: any Sendable]] = [
        ["function": ["name": "write_file"] as [String: any Sendable]]
    ]
    private let call = """
        <tool_call><function=write_file><parameter=content>print(50)</parameter>
        <parameter=path>/app/src/limiter/window.py</parameter></function></tool_call>
        """

    @Test("Prefilled reasoning closes release native calls at every chunk split")
    func closesReleaseTools() {
        for body in ["```python\ndef f():\n", "`unfinished", "\"unfinished"] {
            for close in ["\n</think>\n\n", "</think>", ""] {
                let text = Array(body + close + call)
                for policy in ToolCallRecoveryPolicy.allCases {
                    for split in 0 ... text.count {
                        let processor = ToolCallProcessor(
                            format: .qwen35, tools: tools, toolCallPolicy: .init(recovery: policy),
                            initialReasoning: QwenReasoningProtocol.tagged)
                        let output =
                            processor.processChunkOutputs(String(text[..<split]))
                            + processor.processChunkOutputs(String(text[split...]))
                            + processor.processEOSOutputs()
                        let calls = output.compactMap {
                            if case .toolCall(let call) = $0 { call } else { nil }
                        }
                        #expect(response(output) == body + close)
                        #expect(calls.count == 1)
                        #expect(calls.first?.function.name == "write_file")
                        #expect(
                            calls.first?.function.arguments["content"]
                                == .string("print(50)"))
                        #expect(processor.rejectedToolCallCount == 0)
                        #expect(processor.recoveredToolCallCount == 0)
                    }
                }
            }
        }
    }

    @Test("Answer code examples remain protected after a reasoning close")
    func answerExamplesStayInert() {
        let prefix = "`unfinished\n</think>\n\n```xml\n" + call + "\n</think>\n```\n"
        for primed in [false, true] {
            let processor = ToolCallProcessor(
                format: .qwen35, tools: tools,
                initialReasoning: primed ? QwenReasoningProtocol.tagged : nil)
            let text = (primed ? prefix : "```xml\n" + call + "\n</think>\n```\n")
            let outputs =
                (text + call).flatMap { processor.processChunkOutputs(String($0)) }
                + processor.processEOSOutputs()
            let calls = outputs.compactMap {
                if case .toolCall(let call) = $0 { call } else { nil }
            }
            #expect(response(outputs) == text)
            #expect(calls.count == 1)
            #expect(processor.rejectedToolCallCount == 0)
        }
    }

    @Test("Canonical closes protect reasoning with absent or empty tool declarations")
    func noDeclaredTools() {
        let config = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .alwaysOn)
        for declarations: [[String: any Sendable]]? in [nil, []] {
            let processor = ToolCallProcessor(
                format: .qwen35, tools: declarations, initialReasoning: config)
            let text = "```python\n" + call + "\n</think>\n\nanswer"
            let outputs =
                text.flatMap { processor.processChunkOutputs(String($0)) }
                + processor.processEOSOutputs()
            let calls = outputs.compactMap {
                if case .toolCall(let call) = $0 { call } else { nil }
            }
            #expect(response(outputs) == text)
            #expect(calls.isEmpty)
            #expect(processor.rejectedToolCallCount == 0)
        }
    }

    private func response(_ output: [ToolCallProcessor.Output]) -> String {
        output.compactMap { if case .response(let text) = $0 { text } else { nil } }.joined()
    }
}
