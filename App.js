import React, { useState, useRef, useEffect, useCallback } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [listening, setListening] = useState(false);
  const [online, setOnline] = useState(navigator.onLine);
  const [ttsEnabled] = useState(true);
  const [queue, setQueue] = useState([]);
  const audioRef = useRef(null);
  const messagesEndRef = useRef(null);
  const audioChunksRef = useRef([]);

  const BACKEND_URL = "http://127.0.0.1:5000";

  // ---------------- Online/Offline detection ----------------
  useEffect(() => {
    const updateOnlineStatus = () => setOnline(navigator.onLine);
    window.addEventListener("online", updateOnlineStatus);
    window.addEventListener("offline", updateOnlineStatus);
    return () => {
      window.removeEventListener("online", updateOnlineStatus);
      window.removeEventListener("offline", updateOnlineStatus);
    };
  }, []);

  // ---------------- Chat Helpers ----------------
  const appendMessage = (msg) => {
    setMessages((prev) => [...prev, msg]);
  };

  // ---------------- Auto-scroll ----------------
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ---------------- TTS ----------------
  const speak = (text) => {
    if (!ttsEnabled || !text) return;
    fetch(`${BACKEND_URL}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })
      .then((res) => res.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        if (audioRef.current) {
          audioRef.current.src = url;
          audioRef.current.play();
        }
      })
      .catch((err) => console.error("TTS error:", err));
  };

  // ---------------- sendMessage ----------------
  const sendMessage = useCallback(
    async (text, fromQueue = false) => {
      if (!text) return;
      if (!fromQueue) appendMessage({ text, who: "user" });

      if (!fromQueue && !online) {
        setQueue((prev) => [...prev, text]);
        appendMessage({ text: "⚠️ Message queued until internet is available.", who: "bot" });
        return;
      }

      try {
        const res = await fetch(`${BACKEND_URL}/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text }),
        });
        const data = await res.json();

        // Show transcribed text if available
        if (data.transcribed) appendMessage({ text: data.transcribed, who: "user" });

        // Handle response messages
        if (data.response) {
          appendMessage({
            text: data.response,
            who: "bot",
            opened: data.opened, // undefined for normal messages
          });
          speak(data.response);
        }
      } catch (err) {
        console.error("Error contacting backend:", err);
        appendMessage({ text: "Error contacting backend", who: "bot" });
        speak("Error contacting backend");
        if (!fromQueue) setQueue((prev) => [...prev, text]);
      }
    },
    [online]
  );

  // ---------------- Queue processing ----------------
  useEffect(() => {
    if (online && queue.length > 0) {
      queue.forEach((q) => sendMessage(q, true));
      setQueue([]);
    }
  }, [online, queue, sendMessage]);

  // ---------------- Apps & Jokes ----------------
  const refreshApps = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/refresh_apps`, { method: "POST" });
      const data = await res.json();
      appendMessage({ text: data.response, who: "bot" });
      speak(data.response);
    } catch {
      appendMessage({ text: "Error refreshing apps", who: "bot" });
      speak("Error refreshing apps");
    }
  };

  const tellJoke = () => sendMessage("tell me a joke");

  // ---------------- Voice Recording ----------------
  const handleVoice = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("Microphone not supported!");
      return;
    }

    setListening(true);
    audioChunksRef.current = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);

    recorder.ondataavailable = (e) => audioChunksRef.current.push(e.data);

    recorder.onstop = async () => {
      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("file", blob, "voice.webm");

      try {
        const res = await fetch(`${BACKEND_URL}/voice`, { method: "POST", body: formData });
        const data = await res.json();

        if (data.transcribed) appendMessage({ text: data.transcribed, who: "user" });
        if (data.response)
          appendMessage({
            text: data.response,
            who: "bot",
            opened: data.opened,
          });
        speak(data.response);
      } catch (err) {
        console.error("Voice API error:", err);
        appendMessage({ text: "Error sending voice to backend.", who: "bot" });
      }
      setListening(false);
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 5000);
  };

  // ---------------- Styles ----------------
  const containerStyle = {
    maxWidth: "700px",
    margin: "40px auto",
    fontFamily: "Arial",
    backgroundColor: online ? "#f9f9f9" : "#1a1a1a",
    color: online ? "#000" : "#eee",
    transition: "0.3s",
    padding: "20px",
    borderRadius: "10px",
    boxShadow: online ? "0 0 10px rgba(0,0,0,.1)" : "0 0 15px rgba(255,255,255,.05)",
  };

  // ---------------- Render ----------------
  return (
    <div style={containerStyle}>
      <div style={{ marginBottom: "20px", textAlign: "center" }}>
        <p>Current Mode: <b>{online ? "ONLINE" : "OFFLINE"}</b></p>
      </div>

      <div style={{ padding: "20px", background: online ? "#fff" : "#333", borderRadius: "10px", boxShadow: "0 0 10px rgba(0,0,0,.1)" }}>
        <h2>Assistant Chat</h2>
        <div style={{ maxHeight: "400px", overflowY: "auto", marginBottom: "10px" }}>
          {messages.map((m, i) => {
            const isAction = m.opened !== undefined;
            let bgColor = m.who === "user" ? "#007bff" : "#eee";
            let textColor = m.who === "user" ? "#fff" : online ? "#333" : "#eee";

            if (isAction) {
              bgColor = m.opened ? "#28a745" : "#dc3545"; // green=success, red=fail
              textColor = "#fff";
            }

            return (
              <div key={i} style={{ marginBottom: "10px" }}>
                <div style={{
                  background: bgColor,
                  color: textColor,
                  textAlign: m.who === "user" ? "right" : "left",
                  padding: "8px 12px",
                  borderRadius: "6px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: m.who === "user" ? "flex-end" : "flex-start"
                }}>
                  <span>{m.text}</span>
                  {isAction && <span style={{ marginLeft: "8px", fontWeight: "bold" }}>{m.opened ? "✅" : "❌"}</span>}
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        <div style={{ display: "flex", gap: "10px", marginBottom: "15px" }}>
          <button onClick={refreshApps} style={{ padding: "10px 15px", borderRadius: "6px", border: "none", background: "#28a745", color: "#fff" }}>Refresh Apps</button>
          <button onClick={tellJoke} style={{ padding: "10px 15px", borderRadius: "6px", border: "none", background: "#ffc107", color: "#000" }}>Tell Joke</button>
        </div>
      </div>

      <div style={{ marginTop: "40px", textAlign: "center", padding: "20px", borderRadius: "10px", boxShadow: "0 0 8px rgba(0,0,0,.05)" }}>
        <h2>Voice Assistant</h2>
        <div
          onClick={handleVoice}
          style={{
            width: "120px",
            height: "120px",
            margin: "20px auto",
            borderRadius: "50%",
            background: listening ? "radial-gradient(circle, #28a745, #218838)" : "radial-gradient(circle, #007bff, #0056b3)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "40px",
            color: "#fff",
            cursor: "pointer",
            boxShadow: listening ? "0 0 25px 10px rgba(40,167,69,0.6)" : "0 0 15px 5px rgba(0,123,255,0.5)",
            animation: listening ? "pulse 1.5s infinite" : "none",
          }}
        >🎤</div>
        <p>{listening ? "Listening..." : "Click the mic to talk"}</p>
      </div>

      <audio ref={audioRef} />

      <style>{`
        @keyframes pulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.1); }
          100% { transform: scale(1); }
        }
      `}</style>
    </div>
  );
}

export default App;
