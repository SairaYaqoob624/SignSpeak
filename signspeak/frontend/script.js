
// MediaPipe Hands & Camera global instances
let webcamVideo = null;
let landmarkCanvas = null;
let canvasCtx = null;
let hands = null;
let camera = null;
let isRecording = false;
let recordingLabel = "";

const startBtn   = document.getElementById("startCamera");
const muteBtn    = document.getElementById("muteBtn");
const resetBtn   = document.getElementById("resetBtn");
const speakBtn   = document.getElementById("speakNowBtn");

const videoFeed  = document.getElementById("videoFeed");
const placeholder = document.querySelector(".video-placeholder");
const trainModal = document.getElementById("trainModal");
const openTrainModal = document.getElementById("openTrainModal");
const closeTrainModal = document.getElementById("closeTrainModal");
const startRecordingBtn = document.getElementById("startRecordingBtn");
const trainAINowBtn = document.getElementById("trainAINowBtn");
const newGestureName = document.getElementById("newGestureName");
const recordingProgressWrap = document.getElementById("recordingProgressWrap");
const recordProgressBar = document.getElementById("recordProgressBar");
const framesCount = document.getElementById("framesCount");
const trainingLog = document.getElementById("trainingLog");

const detectedText = document.getElementById("detectedText");
const historyList  = document.getElementById("historyList");

const speechSpeed = document.getElementById("speechSpeed");
const autoSpeak   = document.getElementById("autoSpeak");

let isMuted    = false;
let cameraOn   = false;
let lastGesture = "";
let lastSpoken  = "";
let currentUser = null;
async function checkAuth() {
    try {
        const res = await fetch("/api/me");
        const data = await res.json();
        
        if (!data.logged_in) {
            if (!window.location.pathname.includes("auth.html")) {
                window.location.href = "/auth.html";
            }
            return;
        }
        
        currentUser = data;
        document.getElementById("userProfile").style.display = "flex";
        document.getElementById("displayUsername").textContent = data.username;
        if (data.role === "admin") {
            document.getElementById("openAnalyticsModal").style.display = "flex";
        }
    } catch (e) {
        console.error("Auth check failed");
    }
}

document.addEventListener("DOMContentLoaded", checkAuth);

const logoutBtn = document.getElementById("logoutBtn");
logoutBtn.addEventListener("click", async () => {
    await fetch("/api/logout");
    window.location.href = "/auth.html";
});
function initMediaPipe() {
    webcamVideo = document.getElementById("webcamVideo");
    landmarkCanvas = document.getElementById("landmarkCanvas");
    canvasCtx = landmarkCanvas.getContext("2d");

    if (typeof Hands === 'undefined') {
        alert("Error: MediaPipe Hands library did not load from the CDN. Please check your internet connection and reload the page.");
        return;
    }

    hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
    });

    hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onHandResults);
}

let lastPostTime = 0;
const POST_INTERVAL = 150; // Predict every 150ms

async function onHandResults(results) {
    if (!cameraOn) return;
    
    landmarkCanvas.width = webcamVideo.videoWidth || 640;
    landmarkCanvas.height = webcamVideo.videoHeight || 480;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            // Draw skeleton lines safely
            if (typeof drawConnectors !== 'undefined') {
                const conn = (typeof HAND_CONNECTIONS !== 'undefined') ? HAND_CONNECTIONS : 
                             ((typeof mpHands !== 'undefined' && mpHands.HAND_CONNECTIONS) ? mpHands.HAND_CONNECTIONS : []);
                drawConnectors(canvasCtx, landmarks, conn, {color: '#00FF00', lineWidth: 5});
            }
            
            // Draw red joints dots safely
            if (typeof drawLandmarks !== 'undefined') {
                drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
            }

            const flattenedLandmarks = [];
            landmarks.forEach(lm => {
                flattenedLandmarks.push(lm.x, lm.y, lm.z);
            });

            if (isRecording && flattenedLandmarks.length === 63) {
                try {
                    await fetch("/api/record_landmark", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            landmarks: flattenedLandmarks,
                            label: recordingLabel
                        })
                    });
                } catch (e) {
                    console.error("Error recording landmark:", e);
                }
            } else if (flattenedLandmarks.length === 63) {
                const now = Date.now();
                if (now - lastPostTime > POST_INTERVAL) {
                    lastPostTime = now;
                    try {
                        await fetch("/api/predict", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ landmarks: flattenedLandmarks })
                        });
                    } catch (e) {
                        console.error("Prediction error:", e);
                    }
                }
            }
        }
    } else {
        if (!isRecording) {
            const now = Date.now();
            if (now - lastPostTime > POST_INTERVAL) {
                lastPostTime = now;
                try {
                    await fetch("/api/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ landmarks: [] })
                    });
                } catch (e) {
                    console.error("Clear gesture error:", e);
                }
            }
        }
    }
    canvasCtx.restore();
}

async function startWebcam() {
    if (!hands) {
        initMediaPipe();
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });
        
        webcamVideo.srcObject = stream;
        webcamVideo.style.display = "block";
        landmarkCanvas.style.display = "block";
        
        // Start playing the video
        await webcamVideo.play();
        
        cameraOn = true;
        placeholder.style.display = "none";
        startBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Camera';
        
        // Start native requestAnimationFrame loop
        onFrame();
        return true;
    } catch (err) {
        console.error("Webcam start failed:", err);
        alert("Webcam error: " + err.message);
        stopWebcam();
        return false;
    }
}

async function onFrame() {
    if (cameraOn && webcamVideo && !webcamVideo.paused && !webcamVideo.ended) {
        try {
            await hands.send({ image: webcamVideo });
        } catch (e) {
            console.error("MediaPipe Hands process frame error:", e);
            const statusBox = document.getElementById("detectedText");
            if (statusBox) {
                statusBox.innerHTML = "<span style='color: #ff5555; font-size: 1.2rem;'>MediaPipe Loading Error. Please Check Internet.</span>";
            }
        }
        requestAnimationFrame(onFrame);
    }
}

function stopWebcam() {
    cameraOn = false;
    if (webcamVideo && webcamVideo.srcObject) {
        const tracks = webcamVideo.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        webcamVideo.srcObject = null;
    }
    if (webcamVideo) {
        webcamVideo.style.display = "none";
    }
    if (landmarkCanvas) {
        landmarkCanvas.style.display = "none";
        if (canvasCtx) canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    }
}

startBtn.addEventListener("click", async () => {
    if (!cameraOn) {
        const success = await startWebcam();
        if (success) {
            placeholder.style.display = "none";
            startBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Camera';
        }
    } else {
        stopWebcam();
        placeholder.style.display = "flex";
        startBtn.innerHTML = '<i class="fas fa-camera"></i> Start Camera';
    }
});
function speak(text) {
    if (isMuted || !text) return;
    speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = parseFloat(speechSpeed.value);
    speechSynthesis.speak(utter);
}
function addToHistory(text) {
    if (!text) return;
    if (historyList.querySelector(".empty-history")) {
        historyList.innerHTML = "";
    }
    const item = document.createElement("div");
    item.className = "history-item";
    item.innerHTML = `
        <span class="history-text">${text}</span>
        <span class="history-time">${new Date().toLocaleTimeString()}</span>
    `;
    historyList.prepend(item);
}
document.querySelectorAll(".phrase-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const phrase = btn.dataset.text;
        detectedText.innerText = phrase;
        addToHistory(phrase);
        if (autoSpeak.checked) speak(phrase);
    });
});
setInterval(async () => {
    if (!cameraOn) return;
    try {
        const res  = await fetch("/live_gesture");
        const data = await res.json();
        if (data.gesture) {
            detectedText.innerText = data.gesture;
        }
    } catch (e) {
        console.log("gesture fetch error");
    }
}, 1000);
muteBtn.addEventListener("click", () => {
    isMuted = !isMuted;
    muteBtn.innerHTML = isMuted
        ? '<i class="fas fa-volume-mute"></i> Unmute'
        : '<i class="fas fa-volume-up"></i> Mute Speech';
});
resetBtn.addEventListener("click", () => {
    detectedText.innerText = "Show Gesture Here";
    historyList.innerHTML  = `<div class="empty-history">No gestures detected yet</div>`;
    lastGesture = "";
    speechSynthesis.cancel();
    sentenceWords = [];
    sbSentenceText.textContent = "Words will appear here…";
    sbSentenceText.classList.remove("sb-has-text");
    sbWordChips.innerHTML = "";
    sbCurrentWord.textContent = "—";
    sbCurrentWord.classList.remove("sb-word-active");
});
speakBtn.addEventListener("click", () => {
    if (detectedText.innerText !== "Show Gesture Here") {
        speak(detectedText.innerText);
    }
});

let autoSpeakMode = "smart";   // "off" | "smart" | "instant"
let holdStartTime  = null;
let holdGesture    = "";
const HOLD_DURATION = 2000;    // 2 seconds to trigger Smart mode
const modeOff         = document.getElementById("modeOff");
const modeSmart       = document.getElementById("modeSmart");
const modeInstant     = document.getElementById("modeInstant");
const modeDescText    = document.getElementById("modeDescText");
const holdProgressWrap = document.getElementById("holdProgressWrap");
const holdProgressFill = document.getElementById("holdProgressFill");
const holdTimerText   = document.getElementById("holdTimerText");
const confThreshold   = document.getElementById("confThreshold");
const confValue       = document.getElementById("confValue");
const speakStatusText = document.getElementById("speakStatusText");
const speakStatusBadge = document.getElementById("speakStatusBadge");

const modeDescriptions = {
    off:     "Auto-speak is disabled",
    smart:   "Hold gesture for 2s → auto speak",
    instant: "Speak immediately on detection"
};
[modeOff, modeSmart, modeInstant].forEach(btn => {
    btn.addEventListener("click", () => {
        autoSpeakMode = btn.dataset.mode;
        [modeOff, modeSmart, modeInstant].forEach(b => b.classList.remove("mode-active"));
        btn.classList.add("mode-active");
        modeDescText.textContent = modeDescriptions[autoSpeakMode];
        holdProgressWrap.style.display = autoSpeakMode === "smart" ? "block" : "none";
        holdStartTime = null;
        holdGesture   = "";
        holdProgressFill.style.width = "0%";
        holdTimerText.textContent    = "0.0s";
        const labels = { off: "Auto-Speak Off", smart: "Smart Mode Active", instant: "Instant Mode" };
        const dotClass = { off: "status-off", smart: "status-smart", instant: "status-instant" };
        speakStatusText.textContent = labels[autoSpeakMode];
        const dot = speakStatusBadge.querySelector(".status-dot");
        dot.className = `status-dot ${dotClass[autoSpeakMode]}`;
        autoSpeak.checked = autoSpeakMode !== "off";
    });
});
confThreshold.addEventListener("input", () => {
    confValue.textContent = confThreshold.value + "%";
});
let speakResetTimeout = null;

setInterval(async () => {
    if (!cameraOn || autoSpeakMode === "off") {
        holdStartTime = null;
        holdProgressFill.style.width = "0%";
        holdTimerText.textContent = "0.0s";
        if (!speakResetTimeout) {
            speakResetTimeout = setTimeout(() => {
                lastSpoken = "";
                speakResetTimeout = null;
            }, 4000);
        }
        return;
    }
    
    if (speakResetTimeout) { 
        clearTimeout(speakResetTimeout); 
        speakResetTimeout = null; 
    }

    try {
        const res  = await fetch("/current_gesture");
        const data = await res.json();

        const word = data.gesture || "";
        const conf = data.confidence || 0;
        const minConf = parseInt(confThreshold.value);
        if (word) {
            sbCurrentWord.textContent = word.toUpperCase();
            sbCurrentWord.classList.add("sb-word-active");
        } else {
            sbCurrentWord.textContent = "—";
            sbCurrentWord.classList.remove("sb-word-active");
            holdStartTime = null;
            holdGesture   = "";
            holdProgressFill.style.width = "0%";
            holdTimerText.textContent = "0.0s";
            return;
        }
        if (conf < minConf) {
            holdStartTime = null;
            holdGesture   = "";
            holdProgressFill.style.width = "0%";
            holdTimerText.textContent = "0.0s";
            return;
        }
        if (autoSpeakMode === "instant") {
            if (word !== lastSpoken) {
                lastSpoken = word;
                speak(word);
            }
            return;
        }
        if (autoSpeakMode === "smart") {
            if (word !== holdGesture) {
                holdGesture   = word;
                holdStartTime = Date.now();
            }

            const elapsed = Date.now() - holdStartTime;
            const pct     = Math.min((elapsed / HOLD_DURATION) * 100, 100);

            holdProgressFill.style.width = pct + "%";
            holdTimerText.textContent    = (elapsed / 1000).toFixed(1) + "s";

            if (elapsed >= HOLD_DURATION && word !== lastSpoken) {
                lastSpoken = word;
                speak(word);
                addToHistory("🤖 " + word);
                holdProgressFill.classList.add("hold-spoken");
                setTimeout(() => {
                    holdProgressFill.classList.remove("hold-spoken");
                    holdProgressFill.style.width = "0%";
                    holdTimerText.textContent = "0.0s";
                    holdStartTime = Date.now(); // reset for cooldown
                    holdGesture   = "";
                }, 600);
            }
        }

    } catch (e) {}
}, 400);   // Balanced poll frequency for CPU health

let sentenceWords = [];

const sbCurrentWord = document.getElementById("sbCurrentWord");
const sbSentenceText = document.getElementById("sbSentenceText");
const sbWordChips   = document.getElementById("sbWordChips");
const sbAddWordBtn  = document.getElementById("sbAddWord");
const sbSpeakBtn    = document.getElementById("sbSpeak");
const sbUndoBtn     = document.getElementById("sbUndo");
const sbClearBtn    = document.getElementById("sbClear");
setInterval(async () => {
    if (!cameraOn) return;
    try {
        const res  = await fetch("/current_gesture");
        const data = await res.json();
        if (data.gesture) {
            sbCurrentWord.textContent = data.gesture.toUpperCase();
            sbCurrentWord.classList.add("sb-word-active");
        } else {
            sbCurrentWord.textContent = "—";
            sbCurrentWord.classList.remove("sb-word-active");
        }
    } catch (e) {}
}, 1000); // Polling reduced to save CPU
function updateSentenceDisplay() {
    if (sentenceWords.length === 0) {
        sbSentenceText.textContent = "Words will appear here…";
        sbSentenceText.classList.remove("sb-has-text");
        sbWordChips.innerHTML = "";
        return;
    }

    const sentence = sentenceWords
        .map((w, i) => i === 0
            ? w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()
            : w.toLowerCase())
        .join(" ");

    sbSentenceText.textContent = sentence;
    sbSentenceText.classList.add("sb-has-text");
    sbWordChips.innerHTML = "";
    sentenceWords.forEach((word, idx) => {
        const chip = document.createElement("span");
        chip.className = "sb-chip";
        chip.innerHTML = `${word.toUpperCase()} <span class="sb-chip-remove" data-idx="${idx}">✕</span>`;
        sbWordChips.appendChild(chip);
    });
    sbWordChips.querySelectorAll(".sb-chip-remove").forEach(x => {
        x.addEventListener("click", () => {
            sentenceWords.splice(parseInt(x.dataset.idx), 1);
            updateSentenceDisplay();
        });
    });
}
sbAddWordBtn.addEventListener("click", () => {
    const word = sbCurrentWord.textContent.trim();
    if (!word || word === "—") {
        sbAddWordBtn.classList.add("sb-shake");
        setTimeout(() => sbAddWordBtn.classList.remove("sb-shake"), 500);
        return;
    }
    sentenceWords.push(word);
    updateSentenceDisplay();

    sbAddWordBtn.innerHTML = '<i class="fas fa-check"></i> Added!';
    setTimeout(() => {
        sbAddWordBtn.innerHTML = '<i class="fas fa-plus-circle"></i> Add Word';
    }, 800);
});
sbSpeakBtn.addEventListener("click", () => {
    if (sentenceWords.length === 0) return;
    const sentence = sentenceWords.join(" ");
    speak(sentence);
    addToHistory("📝 " + sentence);
});
sbUndoBtn.addEventListener("click", () => {
    if (sentenceWords.length > 0) {
        sentenceWords.pop();
        updateSentenceDisplay();
    }
});
sbClearBtn.addEventListener("click", () => {
    sentenceWords = [];
    updateSentenceDisplay();
    speechSynthesis.cancel();
});

function addLog(message, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    trainingLog.prepend(entry);
}
openTrainModal.addEventListener("click", () => {
    trainModal.style.display = "flex";
    addLog("Training modal opened. Camera is OFF by default.");
});

closeTrainModal.addEventListener("click", () => {
    trainModal.style.display = "none";
    isRecording = false;
    recordingLabel = "";
});
window.addEventListener("click", (e) => {
    if (e.target === trainModal) {
        trainModal.style.display = "none";
        isRecording = false;
        recordingLabel = "";
    }
});
let recordingInterval = null;

startRecordingBtn.addEventListener("click", async () => {
    const label = newGestureName.value.trim();
    if (!label) {
        alert("Please enter a gesture name first!");
        return;
    }

    if (!cameraOn) {
        addLog("Activating camera for recording...");
        const success = await startWebcam();
        if (success) {
            placeholder.style.display = "none";
            startBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Camera';
            cameraOn = true;
            await new Promise(r => setTimeout(r, 1000));
        } else {
            return;
        }
    }

    try {
        const res = await fetch("/start_recording", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: label })
        });
        const data = await res.json();

        if (data.status === "started") {
            isRecording = true;
            recordingLabel = label;
            addLog(`Started recording landmarks for: ${label}`, "recording");
            recordingProgressWrap.style.display = "block";
            startRecordingBtn.disabled = true;
            newGestureName.disabled = true;
            recordingInterval = setInterval(pollRecordingStatus, 200);
        }
    } catch (e) {
        addLog("Error starting recording: " + e.message, "error");
    }
});

async function pollRecordingStatus() {
    try {
        const res = await fetch("/recording_status");
        const data = await res.json();

        const pct = (data.count / data.max) * 100;
        recordProgressBar.style.width = pct + "%";
        framesCount.textContent = `${data.count}/${data.max}`;

        if (!data.is_recording || data.count >= data.max) {
            clearInterval(recordingInterval);
            isRecording = false;
            recordingLabel = "";
            addLog(`Recording complete! Captured ${data.count} frames.`, "success");
            startRecordingBtn.disabled = false;
            newGestureName.disabled = false;
            startRecordingBtn.innerHTML = '<i class="fas fa-check"></i> Record More';
            trainAINowBtn.disabled = false;
            trainAINowBtn.classList.add("btn-pulse");
            setTimeout(() => trainAINowBtn.classList.remove("btn-pulse"), 2000);
        }
    } catch (e) {
        console.error("Polling error", e);
    }
}
trainAINowBtn.addEventListener("click", async () => {
    trainAINowBtn.disabled = true;
    trainAINowBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training AI...';
    addLog("Retraining AI model with new dataset. Please wait...", "info");

    try {
        const res = await fetch("/train_model_async", { method: "POST" });
        const data = await res.json();

        if (data.status === "success") {
            addLog("🎉 Success! Model retrained and reloaded.", "success");
            trainAINowBtn.innerHTML = '<i class="fas fa-check-circle"></i> Training Done!';
            confettiAnimation(); 
            
            setTimeout(() => {
                trainAINowBtn.innerHTML = '<i class="fas fa-bolt"></i> Retrain AI Model';
                trainAINowBtn.disabled = false;
            }, 3000);
        } else {
            addLog("❌ Training failed: " + data.message, "error");
            trainAINowBtn.disabled = false;
            trainAINowBtn.innerHTML = '<i class="fas fa-bolt"></i> Try Again';
        }
    } catch (e) {
        addLog("Network error during training.", "error");
        trainAINowBtn.disabled = false;
    }
});

function confettiAnimation() {
    const log = document.getElementById("trainingLog");
    log.style.borderColor = "var(--success)";
    setTimeout(() => log.style.borderColor = "rgba(0,0,0,0.4)", 2000);
}

const analyticsModal = document.getElementById("analyticsModal");
const openAnalyticsModal = document.getElementById("openAnalyticsModal");
const closeAnalyticsModal = document.getElementById("closeAnalyticsModal");

let topGesturesChart = null;
let usageHistoryChart = null;
let liveFeedInterval = null;

openAnalyticsModal.addEventListener("click", () => {
    analyticsModal.style.display = "flex";
    switchTab('overview');
});

closeAnalyticsModal.addEventListener("click", () => {
    analyticsModal.style.display = "none";
    clearInterval(liveFeedInterval);
});
document.querySelectorAll(".admin-tab").forEach(tab => {
    tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        switchTab(target);
    });
});

function switchTab(tabName) {
    document.querySelectorAll(".admin-tab").forEach(t => t.classList.remove("active"));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add("active");

    document.querySelectorAll(".admin-tab-content").forEach(c => c.classList.remove("active"));
    document.getElementById(`tab-${tabName}`).classList.add("active");
    if (tabName !== 'live') clearInterval(liveFeedInterval);
    if (tabName === 'overview') fetchAnalyticsData();
    if (tabName === 'users') fetchUsersData();
    if (tabName === 'system') fetchSystemStatus();
    if (tabName === 'live') {
        fetchLiveFeed();
        liveFeedInterval = setInterval(fetchLiveFeed, 3000);
    }
}

async function fetchAnalyticsData() {
    try {
        const summaryRes = await fetch("/api/analytics/summary");
        const summary = await summaryRes.json();
        
        document.getElementById("totalDetections").textContent = summary.total_detections || 0;
        document.getElementById("uniqueGestures").textContent = summary.unique_gestures || 0;
        document.getElementById("topGestureName").textContent = summary.top_gestures[0]?.gesture || "—";

        const statsRes = await fetch("/api/analytics/stats");
        const stats = await statsRes.json();

        renderTopGesturesChart(summary.top_gestures);
        renderUsageHistoryChart(stats);
    } catch (e) {
        console.error("Error fetching analytics:", e);
    }
}

async function fetchUsersData() {
    try {
        const res = await fetch("/api/admin/users");
        const users = await res.json();
        const tbody = document.getElementById("usersTableBody");
        tbody.innerHTML = "";

        users.forEach(u => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${u.username}</td>
                <td><span class="badge-${u.role}">${u.role.toUpperCase()}</span></td>
                <td>${u.detections} detections</td>
                <td>
                    ${u.role !== 'admin' ? `<button class="btn-delete-user" onclick="deleteUser(${u.id})">Delete</button>` : '—'}
                </td>
            `;
            tbody.appendChild(row);
        });
    } catch (e) {}
}

async function deleteUser(userId) {
    if (!confirm("Are you sure? This will delete all user data and logs forever.")) return;
    
    try {
        const res = await fetch("/api/admin/delete_user", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId })
        });
        const data = await res.json();
        if (data.status === "success") {
            fetchUsersData(); // Refresh list
        }
    } catch (e) {}
}

async function fetchSystemStatus() {
    try {
        const res = await fetch("/api/admin/system_status");
        const data = await res.json();
        const grid = document.getElementById("systemStatusGrid");
        
        const cards = [
            { label: "System Status", value: data.status, icon: data.status === "Healthy" ? "dot" : "" },
            { label: "AI Model", value: data.model_name },
            { label: "Model Size", value: data.model_size },
            { label: "Database Size", value: data.db_size },
            { label: "Server Time", value: data.server_time },
            { label: "Runtime", value: "Python " + data.python_version }
        ];

        grid.innerHTML = cards.map(c => `
            <div class="system-card">
                <h4>${c.label}</h4>
                <div class="value">${c.value}</div>
                ${c.icon ? `<div class="status-indicator"><div class="${c.icon}"></div> Online</div>` : ""}
            </div>
        `).join("");
    } catch (e) {}
}

async function fetchLiveFeed() {
    try {
        const res = await fetch("/api/admin/live_feed");
        const logs = await res.json();
        const container = document.getElementById("adminLiveFeed");
        
        if (logs.length === 0) {
            container.innerHTML = '<div class="empty-feed">No recent activity detected...</div>';
            return;
        }

        container.innerHTML = logs.map(log => `
            <div class="feed-item">
                <div class="feed-left">
                    <div class="feed-word">${log.word.toUpperCase()}</div>
                    <div class="feed-user">by ${log.user}</div>
                </div>
                <div class="feed-time">${log.time}</div>
            </div>
        `).join("");
    } catch (e) {}
}

function renderTopGesturesChart(data) {
    const ctx = document.getElementById('topGesturesChart').getContext('2d');
    if (topGesturesChart) topGesturesChart.destroy();
    topGesturesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(item => item.gesture),
            datasets: [{
                label: 'Occurrences',
                data: data.map(item => item.count),
                backgroundColor: 'rgba(76, 201, 240, 0.6)',
                borderColor: '#4cc9f0',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#aaa' } },
                x: { grid: { display: false }, ticks: { color: '#aaa' } }
            }
        }
    });
}

function renderUsageHistoryChart(data) {
    const ctx = document.getElementById('usageHistoryChart').getContext('2d');
    if (usageHistoryChart) usageHistoryChart.destroy();
    usageHistoryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(item => item.hour),
            datasets: [{
                label: 'Detections',
                data: data.map(item => item.count),
                fill: true,
                backgroundColor: 'rgba(67, 97, 238, 0.2)',
                borderColor: '#4361ee',
                tension: 0.4,
                pointBackgroundColor: '#4cc9f0',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#aaa' } },
                x: { grid: { display: false }, ticks: { color: '#aaa' } }
            }
        }
    });
}

