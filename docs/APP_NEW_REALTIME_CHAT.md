# App_new.py - Real-time Log & Chatbot Features

**Date:** 2025-11-27  
**Version:** 4.0 (Enhanced)

---

## New Features

### 1. Real-time Analysis Log

**Feature:** Live log display during analysis execution.

**Implementation:**

```python
# Create a queue for log messages
log_queue = queue.Queue()
log_messages = []

# Custom stdout capture
class LogCapture(io.StringIO):
    def write(self, text):
        if text.strip():
            log_queue.put(text.strip())
        return super().write(text)

# Redirect stdout
old_stdout = sys.stdout
sys.stdout = LogCapture()

# Run analysis in separate thread
analysis_thread = threading.Thread(target=run_analysis_thread)
analysis_thread.start()

# Update log display while running
while analysis_thread.is_alive():
    while not log_queue.empty():
        msg = log_queue.get_nowait()
        log_messages.append(msg)
    
    # Update display (last 50 lines)
    log_placeholder.code('\n'.join(log_messages[-50:]), language="text")
    time.sleep(0.1)  # Update every 100ms
```

**Benefits:**
- See what's happening in real-time
- Monitor Agent A and Agent B activities
- Debug issues more easily
- Better user experience

**Display:**
```
▼ Analysis Log (Real-time)
  [CDDA] Initializing Tool Kit...
  [OK] CDDA Tool Kit ready
  [CDDA] Initializing GraphRAG...
  [OK] GraphRAG initialized
  [AGENT A] Starting orchestration...
  [AGENT A] Reading diagnostic resource...
  [AGENT A] Evaluating signals: UQ=0.847
  [AGENT A] Decision: Invoke counterfactual simulation
  [AGENT A] Compiling ContextObject...
  [AGENT B] Receiving ContextObject...
  [AGENT B] Generating clinical report...
  [OK] Analysis completed
```

---

### 2. Interactive Chatbot with Agent B

**Feature:** Ask questions about the analysis and get answers from Agent B.

**Architecture:**
```
User Question
    ↓
Diagnostic Context (from analysis result)
    ↓
Agent B (Llama3.1-Aloe-Beta-8B)
    ↓
Clinical Answer
```

**Context Provided to Agent B:**
```python
context_summary = f"""
DIAGNOSTIC CONTEXT:
- Subject: {result.subject_id}
- Prediction: {result.prediction}
- Confidence: {result.confidence:.3f}
- Uncertainty: {result.uq_score:.3f}
- Decision Mode: {result.agent_decision}

CLINICAL REPORT SUMMARY:
{result.clinical_report[:500]}...

EXECUTIVE SUMMARY:
- Headline: {summary.get('headline')}
- Risk Level: {summary.get('risk_level')}
- Key Findings: {summary.get('key_findings')}

PHYSICIAN QUESTION:
{user_question}
"""
```

**UI Components:**
```
Ask Agent B (Clinical Consultant)
├── Chat History
│   ├── You: What are the key risk factors?
│   ├── Agent B: Based on the analysis, the primary risk factors are...
│   ├── You: Should we order additional tests?
│   └── Agent B: Given the high uncertainty (UQ: 0.847), I recommend...
├── Text Input: "Your question:"
└── Buttons: [Send] [Clear Chat]
```

**Example Questions:**
- "What are the key risk factors for this patient?"
- "Should we order additional tests?"
- "How confident are you in this diagnosis?"
- "What follow-up would you recommend?"
- "Can you explain the hippocampal findings?"
- "Is there evidence of mixed pathology?"

**Example Conversation:**
```
You: What are the key risk factors for this patient?

Agent B: Based on the diagnostic analysis, the primary risk factors are:

1. Bilateral hippocampal atrophy (Z-scores: -3.21 and -3.08)
2. Entorhinal cortex volume loss (Z-score: -2.87)
3. High uncertainty score (0.847) suggesting prediction sensitivity

The counterfactual analysis shows that masking these hippocampal 
features reduces AD confidence by 13.2%, confirming they are the 
primary diagnostic drivers.

---

You: Should we order additional tests?

Agent B: Yes, I recommend additional validation due to:

1. High uncertainty (UQ: 0.847) indicates the model is not fully 
   confident in this prediction
2. The counterfactual analysis shows significant feature dependency
3. Clinical correlation is strongly recommended

Suggested tests:
- PET imaging (amyloid or tau)
- CSF biomarkers (Aβ42, tau, p-tau)
- Cognitive assessment battery
- Follow-up MRI in 6-12 months to track progression
```

---

## Technical Details

### Real-time Log Implementation

**Threading:**
```python
def run_analysis_thread():
    try:
        result_container['result'] = agent.run_analysis(selected_subject)
    except Exception as e:
        error_container['error'] = e

analysis_thread = threading.Thread(target=run_analysis_thread)
analysis_thread.start()
```

**Log Capture:**
```python
class LogCapture(io.StringIO):
    def write(self, text):
        if text.strip():
            log_queue.put(text.strip())
        return super().write(text)
```

**Display Update:**
```python
while analysis_thread.is_alive():
    # Get new messages
    while not log_queue.empty():
        msg = log_queue.get_nowait()
        log_messages.append(msg)
    
    # Update display (last 50 lines to avoid overflow)
    log_placeholder.code('\n'.join(log_messages[-50:]), language="text")
    
    time.sleep(0.1)  # 100ms update interval
```

### Chatbot Implementation

**Agent B Initialization:**
```python
if 'agent_b_chat' not in st.session_state:
    config = AgentBConfig(
        model="llama3.1-aloe-beta-8b",
        model_path=consultant_path,
        provider="huggingface",
        temperature=0.3,
        use_llm=use_llm,
        load_in_8bit=not use_4bit,
        verbose=False
    )
    st.session_state.agent_b_chat = AgentB(config=config)
```

**Response Generation:**
```python
response = agent_b.llm_provider.handle_text(
    prompt=chat_prompt,
    model_path=agent_b.config.model_path,
    system_instruction="You are a clinical consultant AI. Provide clear, evidence-based answers."
)
```

**Chat History Management:**
```python
# Initialize
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Add messages
st.session_state.chat_history.append(("user", user_question))
st.session_state.chat_history.append(("agent", response))

# Display
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Agent B:** {message}")
```

---

## Performance Impact

| Feature | Impact | Notes |
|---------|--------|-------|
| Real-time Log | +0.1-0.2s | Threading overhead |
| Log Display | Minimal | Updates every 100ms |
| Chatbot | +3-8s per question | LLM inference time |
| Chat History | Minimal | Stored in session state |

---

## User Experience

### Analysis Flow with Real-time Log

```
1. Click "Start Analysis"
   ↓
2. See initialization progress
   [CDDA] Initializing Tool Kit...
   [OK] CDDA Tool Kit ready
   ↓
3. See Agent A activities
   [AGENT A] Reading diagnostic resource...
   [AGENT A] Evaluating signals...
   [AGENT A] Decision: Invoke counterfactual
   ↓
4. See Agent B activities
   [AGENT B] Receiving ContextObject...
   [AGENT B] Generating clinical report...
   ↓
5. Analysis complete
   [OK] Analysis completed (10.3s)
```

### Chatbot Interaction Flow

```
1. Review analysis results
   ↓
2. Scroll to "Ask Agent B" section
   ↓
3. Type question
   ↓
4. Click "Send"
   ↓
5. See "Agent B is thinking..." spinner
   ↓
6. Read Agent B's response
   ↓
7. Ask follow-up questions
   ↓
8. Clear chat if starting new topic
```

---

## Benefits

### For Clinicians

1. **Transparency**: See exactly what the system is doing
2. **Confidence**: Real-time feedback builds trust
3. **Interaction**: Ask questions to clarify findings
4. **Education**: Learn from Agent B's explanations
5. **Efficiency**: Get answers without leaving the interface

### For Researchers

1. **Debugging**: Identify issues in real-time
2. **Monitoring**: Track agent activities
3. **Validation**: Verify system behavior
4. **Documentation**: Log captures complete execution trace

### For System

1. **Feedback**: User questions improve understanding of needs
2. **Validation**: Chatbot responses can be reviewed for quality
3. **Engagement**: Interactive features increase user satisfaction

---

## Limitations

### Real-time Log

1. **Buffer Size**: Limited to last 50 lines to prevent overflow
2. **Update Rate**: 100ms interval may miss very fast events
3. **Threading**: May not capture all stdout in complex scenarios

### Chatbot

1. **Context Window**: Limited to summary (not full report)
2. **Response Time**: 3-8 seconds per question
3. **No Memory**: Each question is independent (no conversation memory)
4. **LLM Required**: Only works when LLM mode is enabled

---

## Future Enhancements

### Real-time Log

1. **Log Levels**: Filter by INFO/WARNING/ERROR
2. **Search**: Search within logs
3. **Export**: Download logs as text file
4. **Timestamps**: Add timestamps to each log entry

### Chatbot

2. **Conversation Memory**: Remember previous questions in session
3. **Suggested Questions**: Show common questions as buttons
4. **Voice Input**: Speech-to-text for questions
5. **Export Chat**: Download conversation as PDF
6. **Multi-turn Context**: Maintain conversation context

---

## Testing Checklist

- [x] Real-time log displays during analysis
- [x] Log updates every 100ms
- [x] Log shows Agent A and Agent B activities
- [x] Chatbot accepts user questions
- [x] Agent B provides relevant answers
- [x] Chat history persists during session
- [x] Clear chat button works
- [x] No UI blocking during chat
- [x] Error handling for failed responses

---

## Known Issues

1. **Log Capture**: May not capture all output if agents use different logging mechanisms
2. **Chat Context**: Limited to 500 characters of clinical report
3. **Response Quality**: Depends on Agent B's LLM quality

---

**Document Version:** 4.0  
**Last Updated:** 2025-11-27  
**Status:** ✅ Complete  
**Author:** Development Team
