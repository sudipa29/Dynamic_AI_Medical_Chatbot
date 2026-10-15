
/* ============================================================
   Dynamic AI Chatbot — three-page dashboard frontend
   Compatible with overview.html, assistant.html, insights.html
   API: GET /api/health and POST /api/chat

   Behavior:
   - Hard refresh: clear displayed conversation messages.
   - Dashboard navigation: preserve conversation messages.
   - New Chat: clear messages, preserve accumulated analytics.
   - Analytics and latest insights survive a hard refresh.
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const page = document.body.dataset.page || "assistant";
    const STORAGE_KEY = "zidioChatbotState";
    const LEGACY_KEY = "dynamicAIConversations";

    const defaultState = () => ({
        messages: [],
        analytics: [],
        conversationCount: 1,
        latest: null
    });

    // --------------------------------------------------------
    // 1. DETECT A BROWSER PAGE RELOAD
    // --------------------------------------------------------

    function isPageReload() {
        try {
            const navigationEntries =
                performance.getEntriesByType("navigation");

            if (navigationEntries.length > 0) {
                return navigationEntries[0].type === "reload";
            }

            // Fallback for older browsers.
            return (
                performance.navigation &&
                performance.navigation.type === 1
            );
        } catch (error) {
            console.warn(
                "Could not determine navigation type:",
                error
            );

            return false;
        }
    }

    const pageWasReloaded = isPageReload();

    // --------------------------------------------------------
    // 2. RESTORE PREVIOUS CONVERSATION AND ANALYTICS
    // --------------------------------------------------------

    function readSavedState() {
        for (const storage of [
            window.sessionStorage,
            window.localStorage
        ]) {
            try {
                const raw = storage.getItem(STORAGE_KEY);

                if (!raw) continue;

                const parsed = JSON.parse(raw);

                if (parsed && typeof parsed === "object") {
                    return {
                        ...defaultState(),
                        ...parsed,

                        messages: Array.isArray(parsed.messages)
                            ? parsed.messages
                            : [],

                        analytics: Array.isArray(parsed.analytics)
                            ? parsed.analytics
                            : [],

                        conversationCount:
                            Number(parsed.conversationCount) || 1,

                        latest: parsed.latest || null
                    };
                }
            } catch (error) {
                console.warn(
                    "Could not restore saved chatbot state:",
                    error
                );
            }
        }

        // Compatibility with the older conversation storage format.
        try {
            const legacy = localStorage.getItem(LEGACY_KEY);

            if (legacy) {
                const parsed = JSON.parse(legacy);

                if (Array.isArray(parsed)) {
                    const messages = parsed
                        .flatMap((conversation) => {
                            if (Array.isArray(conversation.messages)) {
                                return conversation.messages;
                            }

                            if (Array.isArray(conversation.history)) {
                                return conversation.history;
                            }

                            return [];
                        })
                        .filter(
                            (message) =>
                                message &&
                                (message.role || message.sender) &&
                                (message.content || message.text)
                        )
                        .map((message) => ({
                            role: /user/i.test(
                                String(message.role || message.sender)
                            )
                                ? "user"
                                : "assistant",

                            content: String(
                                message.content || message.text
                            )
                        }));

                    return {
                        ...defaultState(),
                        messages
                    };
                }
            }
        } catch (error) {
            console.warn(
                "Could not read legacy conversation history:",
                error
            );
        }

        return defaultState();
    }

    const state = readSavedState();
    let sending = false;

    function saveState() {
        const serialized = JSON.stringify(state);

        try {
            localStorage.setItem(STORAGE_KEY, serialized);
        } catch (error) {
            console.warn(
                "Could not save state to localStorage:",
                error
            );
        }

        try {
            sessionStorage.setItem(STORAGE_KEY, serialized);
        } catch (error) {
            console.warn(
                "Could not save state to sessionStorage:",
                error
            );
        }
    }

    // On a reload, clear only the current conversation.
    // Keep analytics, conversation count, and latest insights.
    if (pageWasReloaded) {
        state.messages = [];
        saveState();

        console.info(
            "Page reloaded: current chat cleared; analytics preserved."
        );
    }

    // --------------------------------------------------------
    // 3. GENERAL HELPERS
    // --------------------------------------------------------

    function text(value, fallback = "—") {
        if (
            value === null ||
            value === undefined ||
            typeof value === "object"
        ) {
            return fallback;
        }

        const result = String(value).trim();

        return result || fallback;
    }

    function label(value) {
        return text(value, "Unknown")
            .replace(/[_-]+/g, " ")
            .replace(/\b\w/g, (letter) => letter.toUpperCase());
    }

    function confidence(value) {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return null;
        }

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return null;
        }

        const percentageValue =
            number >= 0 && number <= 1
                ? number * 100
                : number;

        return Math.max(0, Math.min(100, percentageValue));
    }

    function percentage(value) {
        const result = confidence(value);

        return result === null
            ? "—"
            : `${result.toFixed(1)}%`;
    }

    function duration(value) {
        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return "—";
        }

        const number = Number(value);

        if (!Number.isFinite(number) || number < 0) {
            return "—";
        }

        return number < 1000
            ? `${Math.round(number)} ms`
            : `${(number / 1000).toFixed(2)} s`;
    }

    function sentiment(value) {
        if (value && typeof value === "object") {
            value =
                value.label ||
                value.sentiment ||
                value.polarity ||
                value.classification ||
                "";
        }

        const result = text(value, "").toLowerCase();

        if (result.includes("positive")) return "positive";
        if (result.includes("negative")) return "negative";
        if (result.includes("neutral")) return "neutral";

        return "unknown";
    }

    function setStatus(status, isOnline) {
        if ($("connectionStatus")) {
            $("connectionStatus").textContent = status;
        }

        if ($("statusIndicator")) {
            $("statusIndicator").classList.toggle(
                "online",
                Boolean(isOnline)
            );

            $("statusIndicator").classList.toggle(
                "offline",
                !isOnline
            );
        }
    }

    async function checkHealth() {
        try {
            const response = await fetch("/api/health", {
                headers: {
                    Accept: "application/json"
                }
            });

            if (!response.ok) {
                throw new Error(
                    `Health endpoint returned HTTP ${response.status}`
                );
            }

            const data = await response.json();

            const status = text(
                data.api_status || data.status,
                "Online"
            );

            setStatus(
                status,
                /online|ok|success|healthy/i.test(status)
            );
        } catch (error) {
            console.warn("API health check failed:", error);
            setStatus("API unavailable", false);
        }
    }

    // --------------------------------------------------------
    // 4. OVERVIEW PAGE
    // --------------------------------------------------------

    function renderOverview() {
        if (!$("totalMessages")) return;

        $("totalMessages").textContent =
            String(state.analytics.length);

        $("totalConversations").textContent =
            String(Math.max(1, state.conversationCount));

        const confidenceValues = state.analytics
            .map((item) => confidence(item.confidence))
            .filter((number) => number !== null);

        $("averageConfidence").textContent =
            confidenceValues.length
                ? percentage(
                    confidenceValues.reduce(
                        (sum, number) => sum + number,
                        0
                    ) / confidenceValues.length
                )
                : "—";

        const responseTimes = state.analytics
            .map((item) => Number(item.responseTime))
            .filter(
                (number) =>
                    Number.isFinite(number) && number >= 0
            );

        $("averageResponseTime").textContent =
            responseTimes.length
                ? duration(
                    responseTimes.reduce(
                        (sum, number) => sum + number,
                        0
                    ) / responseTimes.length
                )
                : "—";

        // Intent distribution.
        const chart = $("intentChart");

        if (chart) {
            chart.replaceChildren();

            const counts = {};

            state.analytics.forEach((item) => {
                const intent = text(item.intent, "unknown");

                counts[intent] = (counts[intent] || 0) + 1;
            });

            const entries = Object.entries(counts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8);

            if (!entries.length) {
                chart.textContent =
                    "Send a message in AI Assistant to populate this chart.";
            } else {
                const max = Math.max(
                    ...entries.map((entry) => entry[1])
                );

                entries.forEach(([intent, count]) => {
                    const row = document.createElement("div");
                    row.className = "bar-row";

                    const name = document.createElement("span");
                    name.className = "bar-label";
                    name.textContent = label(intent);

                    const track = document.createElement("div");
                    track.className = "bar-track";

                    const fill = document.createElement("div");
                    fill.className = "bar-fill";
                    fill.style.width =
                        `${(count / max) * 100}%`;

                    track.appendChild(fill);

                    const number = document.createElement("b");
                    number.textContent = String(count);

                    row.append(name, track, number);
                    chart.appendChild(row);
                });
            }
        }

        // Sentiment distribution.
        const ring = $("sentimentRing");

        if (ring) {
            const counts = {
                positive: 0,
                neutral: 0,
                negative: 0
            };

            state.analytics.forEach((item) => {
                const tone = sentiment(item.sentiment);

                if (Object.hasOwn(counts, tone)) {
                    counts[tone] += 1;
                }
            });

            const total =
                counts.positive +
                counts.neutral +
                counts.negative;

            const positiveEnd = total
                ? (counts.positive / total) * 100
                : 0;

            const neutralEnd = total
                ? ((counts.positive + counts.neutral) / total) * 100
                : 100;

            ring.style.background =
                `conic-gradient(
                    #28b487 0 ${positiveEnd}%,
                    #e8b45e ${positiveEnd}% ${neutralEnd}%,
                    #e47780 ${neutralEnd}% 100%
                )`;

            if ($("sentimentCount")) {
                $("sentimentCount").textContent = String(total);
            }

            if ($("positiveCount")) {
                $("positiveCount").textContent =
                    String(counts.positive);
            }

            if ($("neutralCount")) {
                $("neutralCount").textContent =
                    String(counts.neutral);
            }

            if ($("negativeCount")) {
                $("negativeCount").textContent =
                    String(counts.negative);
            }

            if ($("sentimentNote")) {
                $("sentimentNote").textContent = total
                    ? `${total} messages analyzed.`
                    : "Statistics appear after successful messages.";
            }
        }

        // Recent activity table.
        const table = $("activityTableBody");

        if (table) {
            table.replaceChildren();

            const recent = [...state.analytics]
                .reverse()
                .slice(0, 8);

            if (!recent.length) {
                const row = document.createElement("tr");
                const cell = document.createElement("td");

                cell.colSpan = 4;
                cell.textContent =
                    "No activity yet. Send a message in AI Assistant to see it here.";

                row.appendChild(cell);
                table.appendChild(row);
            } else {
                recent.forEach((item) => {
                    const row = document.createElement("tr");

                    [
                        item.message,
                        label(item.intent),
                        percentage(item.confidence),
                        item.time
                    ].forEach((value) => {
                        const cell = document.createElement("td");
                        cell.textContent = text(value);
                        row.appendChild(cell);
                    });

                    table.appendChild(row);
                });
            }
        }
    }

    // --------------------------------------------------------
    // 5. CHAT MESSAGE RENDERING AND RESTORATION
    // --------------------------------------------------------

    function renderMessage(role, content, timestamp) {
        const box = $("chatMessages");

        if (!box) return;

        const row = document.createElement("div");
        row.className =
            `msg ${role === "user" ? "user" : "bot"}`;

        const icon = document.createElement("span");
        icon.className = "msg-icon";
        icon.textContent = role === "user" ? "U" : "✳";

        const wrap = document.createElement("div");

        const meta = document.createElement("small");
        meta.textContent =
            `${role === "user" ? "You" : "Medical AI"} · ` +
            (timestamp ||
                new Date().toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit"
                }));

        const bubble = document.createElement("div");
        bubble.className = "bubble";

        // Render inline Markdown safely using DOM elements.
        function appendInlineMarkdown(parent, value) {
            const pattern =
                /(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`)/g;

            let lastIndex = 0;
            let match;

            while ((match = pattern.exec(value)) !== null) {
                if (match.index > lastIndex) {
                    parent.appendChild(
                        document.createTextNode(
                            value.slice(lastIndex, match.index)
                        )
                    );
                }

                const token = match[0];

                const isBold =
                    token.startsWith("**") ||
                    token.startsWith("__");

                const isCode = token.startsWith("`");

                const element = document.createElement(
                    isCode ? "code" : isBold ? "strong" : "em"
                );

                element.textContent = token.slice(
                    isBold ? 2 : 1,
                    isBold ? -2 : -1
                );

                parent.appendChild(element);
                lastIndex = pattern.lastIndex;
            }

            if (lastIndex < value.length) {
                parent.appendChild(
                    document.createTextNode(value.slice(lastIndex))
                );
            }
        }

        // Render headings, paragraphs, and ordered/unordered lists.
        function appendBlock(parent, block) {
            const lines = block.split("\n");
            let list = null;
            let listType = null;

            function closeList() {
                list = null;
                listType = null;
            }

            lines.forEach((line) => {
                const heading = line.match(
                    /^\s{0,3}(#{1,6})\s+(.+)$/
                );

                const bullet = line.match(
                    /^\s*[-*+]\s+(.+)$/
                );

                const numbered = line.match(
                    /^\s*\d+[.)]\s+(.+)$/
                );

                if (heading) {
                    closeList();

                    const element = document.createElement(
                        `h${heading[1].length}`
                    );

                    appendInlineMarkdown(element, heading[2]);
                    parent.appendChild(element);
                    return;
                }

                if (bullet || numbered) {
                    const type = bullet ? "ul" : "ol";

                    if (!list || listType !== type) {
                        closeList();
                        list = document.createElement(type);
                        listType = type;
                        parent.appendChild(list);
                    }

                    const item = document.createElement("li");

                    appendInlineMarkdown(
                        item,
                        (bullet || numbered)[1]
                    );

                    list.appendChild(item);
                    return;
                }

                closeList();

                if (!line.trim()) return;

                const paragraph = document.createElement("p");
                appendInlineMarkdown(paragraph, line);
                parent.appendChild(paragraph);
            });
        }

        const normalizedContent = String(content ?? "")
            .replace(/\r\n?/g, "\n")
            .trim();

        normalizedContent
            .split(/\n\s*\n/)
            .filter((block) => block.trim())
            .forEach((block) => appendBlock(bubble, block));

        wrap.append(meta, bubble);

        if (role === "user") {
            row.append(wrap, icon);
        } else {
            row.append(icon, wrap);
        }

        box.appendChild(row);
        box.scrollTop = box.scrollHeight;
    }

    function restoreChat() {
        const box = $("chatMessages");

        if (!box) return;

        box.replaceChildren();

        if (!state.messages.length) {
            renderMessage(
                "assistant",
                "Welcome to the Medical AI Assistant.\n\n" +
                "I can help you explore general health information, " +
                "medical terminology, common symptoms and preventive care.\n\n" +
                "What would you like to learn about today?"
            );

            return;
        }

        state.messages.forEach((message) => {
            renderMessage(
                message.role,
                message.content,
                message.time
            );
        });
    }

    function resizeInput() {
        const input = $("messageInput");

        if (!input) return;

        input.style.height = "auto";
        input.style.height =
            `${Math.min(input.scrollHeight, 140)}px`;

        if ($("characterCount")) {
            $("characterCount").textContent =
                `${input.value.length} / 4000`;
        }
    }

    // --------------------------------------------------------
    // 6. CONVERSATION INSIGHTS PAGE
    // --------------------------------------------------------

    function renderInsights() {
        if (!$("insightsContent") || !$("noInsights")) return;

        const data = state.latest;

        if (!data) {
            $("noInsights").classList.remove("hidden");
            $("insightsContent").classList.add("hidden");
            return;
        }

        $("noInsights").classList.add("hidden");
        $("insightsContent").classList.remove("hidden");

        const confidenceValue = confidence(data.confidence);
        const tone = sentiment(data.sentiment);

        $("insightIntent").textContent = label(data.intent);

        $("insightTopic").textContent =
            text(data.topic, "Topic not available");

        $("insightConfidence").textContent =
            percentage(data.confidence);

        $("confidenceBar").style.width =
            `${confidenceValue || 0}%`;

        $("insightSentiment").textContent = label(tone);

        $("insightSentiment").className =
            tone === "unknown" ? "" : `${tone}-text`;

        $("sentimentDescription").textContent =
            tone === "unknown"
                ? "Sentiment unavailable"
                : `${label(tone)} tone detected`;

        $("insightResponseTime").textContent =
            duration(data.response_time_ms);

        $("latestMessage").textContent =
            text(data.message, "Message not available");

        $("latestTime").textContent =
            text(data.time, "—");

        $("insightApiStatus").textContent =
            text(data.api_status, "Online");

        $("insightSecondIntent").textContent =
            label(data.second_intent);

        $("insightSecondConfidence").textContent =
            percentage(data.second_confidence);

        const margin = confidence(data.confidence_margin);

            $("insightMargin").textContent =
            data.confidence_margin === null ||
            data.confidence_margin === undefined ||
            data.confidence_margin === ""
                ? "—"
                : margin !== null
                ? `${margin.toFixed(1)} percentage points`
                : "—";

        const entityContainer = $("insightEntities");

        if (entityContainer) {
            entityContainer.replaceChildren();

            let entities =
                data.entities ||
                data.extracted_entities ||
                data.named_entities ||
                [];

            if (
                !Array.isArray(entities) &&
                entities &&
                typeof entities === "object"
            ) {
                entities = Object.entries(entities).flatMap(
                    ([kind, values]) =>
                        (Array.isArray(values) ? values : [values])
                            .map((value) => ({
                                type: kind,
                                text: value
                            }))
                );
            }

            if (Array.isArray(entities)) {
                entities.forEach((entity) => {
                    const value =
                        typeof entity === "string"
                            ? entity
                            : text(
                                entity?.text ||
                                entity?.value ||
                                entity?.entity ||
                                entity?.word,
                                ""
                            );

                    if (!value || value === "—") return;

                    const tag = document.createElement("span");
                    tag.className = "entity-tag";

                    tag.textContent =
                        `${typeof entity === "object"
                            ? text(
                                entity.label ||
                                entity.type ||
                                entity.entity_type,
                                "Entity"
                            )
                            : "Entity"}: ${value}`;

                    entityContainer.appendChild(tag);
                });
            }

            if (!entityContainer.children.length) {
                entityContainer.textContent =
                    "No entities returned for this message.";
            }
        }
    }

    // --------------------------------------------------------
    // 7. SEND A MESSAGE TO THE EXISTING FLASK API
    // --------------------------------------------------------

    async function sendMessage(rawMessage) {
        const message = String(rawMessage || "").trim();

        if (!message || sending) return;
        if (!$("messageInput") || !$("chatMessages")) return;

        if ($("chatError")) {
            $("chatError").textContent = "";
            $("chatError").classList.add("hidden");
        }

        const userTime =
            new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit"
            });

        state.messages.push({
            role: "user",
            content: message,
            time: userTime
        });

        renderMessage("user", message, userTime);

        $("messageInput").value = "";
        resizeInput();

        sending = true;

        if ($("sendButton")) {
            $("sendButton").disabled = true;
        }

        if ($("sendButtonText")) {
            $("sendButtonText").textContent = "Sending…";
        }

        if ($("typingIndicator")) {
            $("typingIndicator").classList.remove("hidden");
        }

        if ($("messageStatus")) {
            $("messageStatus").textContent =
                "Assistant is preparing a response…";
        }

        const started = performance.now();

        try {
            const history = state.messages
                .slice(-30)
                .map(
                    (item) =>
                        `${item.role === "user" ? "User" : "Assistant"}: ${item.content}`
                )
                .join("\n");

            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json"
                },
                body: JSON.stringify({
                    message,
                    conversation_history: history
                })
            });

            const raw = await response.text();

            let data;

            try {
                data = JSON.parse(raw);
            } catch {
                throw new Error(
                    `Server returned a non-JSON response (HTTP ${response.status}). Check the Flask console.`
                );
            }

            if (!response.ok || data.success === false) {
                throw new Error(
                    text(
                        data.error || data.message,
                        `Request failed (HTTP ${response.status}).`
                    )
                );
            }

            const answer = text(
                data.response || data.answer || data.reply,
                ""
            );

            if (!answer) {
                throw new Error(
                    "The API response did not include a response field."
                );
            }

            const elapsed = performance.now() - started;

            const responseTime =
                data.response_time_ms !== undefined &&
                Number.isFinite(Number(data.response_time_ms))
                    ? Number(data.response_time_ms)
                    : elapsed;

            const intent = text(
                data.intent || data.predicted_intent,
                "unknown"
            );

            const tone = sentiment(data.sentiment);

            const assistantTime =
                new Date().toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit"
                });

            state.analytics.push({
                message,
                intent,
                confidence: data.confidence,
                sentiment: tone,
                responseTime,

                time: new Date().toLocaleString([], {
                    month: "short",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit"
                })
            });

            state.messages.push({
                role: "assistant",
                content: answer,
                time: assistantTime
            });

            state.latest = {
                ...data,
                message,
                response_time_ms: responseTime,
                time: new Date().toLocaleString()
            };

            // Save the conversation and analytics.
            saveState();

            renderMessage("assistant", answer, assistantTime);
            renderOverview();
            renderInsights();

            setStatus(text(data.api_status, "Online"), true);

            if ($("messageStatus")) {
                $("messageStatus").textContent = "Response received";
            }
        } catch (error) {
            console.error("Chat API request failed:", error);

            const errorMessage = text(
                error.message,
                "Unexpected error"
            );

            if ($("chatError")) {
                $("chatError").textContent = errorMessage;
                $("chatError").classList.remove("hidden");
            }

            if ($("messageStatus")) {
                $("messageStatus").textContent = "Request failed";
            }

            setStatus("API request failed", false);

            // Preserve the user's message if the API fails.
            saveState();
        } finally {
            sending = false;

            if ($("sendButton")) {
                $("sendButton").disabled = false;
            }

            if ($("sendButtonText")) {
                $("sendButtonText").textContent = "Send";
            }

            if ($("typingIndicator")) {
                $("typingIndicator").classList.add("hidden");
            }

            if ($("messageInput")) {
                $("messageInput").focus();
            }
        }
    }

    // --------------------------------------------------------
    // 8. NEW CONVERSATION
    // --------------------------------------------------------

    function startNewConversation() {
        if (sending) return;

        state.conversationCount =
            Math.max(1, Number(state.conversationCount) || 1) + 1;

        state.messages = [];
        state.latest = null;

        // Preserve analytics from earlier conversations.
        saveState();

        restoreChat();
        renderInsights();

        if ($("messageInput")) {
            $("messageInput").value = "";
            resizeInput();
            $("messageInput").focus();
        }

        if ($("messageStatus")) {
            $("messageStatus").textContent =
                "New conversation started";
        }

        renderOverview();
    }

    // --------------------------------------------------------
    // 9. NAVIGATION — ONLY ONE ACTIVE PAGE
    // --------------------------------------------------------

    document
        .querySelectorAll(".topbar nav a[data-link]")
        .forEach((link) => {
            const isCurrent = link.dataset.link === page;

            link.classList.toggle("active", isCurrent);

            if (isCurrent) {
                link.setAttribute("aria-current", "page");
            } else {
                link.removeAttribute("aria-current");
            }
        });

    // --------------------------------------------------------
    // 10. ASSISTANT PAGE EVENTS
    // --------------------------------------------------------

    if (page === "assistant") {
        restoreChat();

        if ($("chatForm")) {
            $("chatForm").addEventListener("submit", (event) => {
                event.preventDefault();
                sendMessage($("messageInput")?.value);
            });
        }

        if ($("messageInput")) {
            $("messageInput").addEventListener("input", resizeInput);

            $("messageInput").addEventListener("keydown", (event) => {
                if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.isComposing
                ) {
                    event.preventDefault();

                    if (!sending && $("chatForm")) {
                        $("chatForm").requestSubmit();
                    }
                }
            });
        }

        if ($("newChatButton")) {
            $("newChatButton").addEventListener(
                "click",
                startNewConversation
            );
        }

        // Medical quick prompts use data-message attributes.
        document
            .querySelectorAll(
                ".suggestion, .suggestion-chip, .text-action"
            )
            .forEach((button) => {
                button.addEventListener("click", () => {
                    if (!sending && button.dataset.message) {
                        sendMessage(button.dataset.message);
                    }
                });
            });

        resizeInput();
    }

    // --------------------------------------------------------
    // 11. INITIALIZE ALL THREE PAGES
    // --------------------------------------------------------

    renderOverview();
    renderInsights();
    checkHealth();

    console.info(
        `Dynamic AI Chatbot initialized on the ${page} page.`
    );
});
