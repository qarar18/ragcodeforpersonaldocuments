import React, { useState } from "react";
import axios from "axios";

function App() {
  const [url, setUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const loadUrl = async () => {
    try {
      await axios.post("http://localhost:8000/load_url", { url });
      alert(" URL content loaded!");
    } catch (err) {
      console.error("Error loading URL:", err);
    }
  };

  const ask = async () => {
    try {
      const res = await axios.post("http://localhost:8000/ask", { question });
      setAnswer(res.data.answer);
    } catch (err) {
      console.error("Error asking question:", err);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.heading}>🔍 RAG Assistant</h1>

        <input
          type="text"
          placeholder="🔗 Enter URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={styles.input}
        />
        <button onClick={loadUrl} style={styles.button}>
          🚀 Load URL
        </button>

        <input
          type="text"
          placeholder="❓ Ask a question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={styles.input}
        />
        <button onClick={ask} style={styles.button}>
           Ask
        </button>

        <h2 style={{ marginTop: "2rem", color: "#333" }}>🧠 Answer:</h2>
        <p style={styles.answer}>{answer}</p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    background: "#f0f4f8",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "2rem",
  },
  card: {
    background: "#fff",
    padding: "2rem",
    borderRadius: "1rem",
    boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
    width: "100%",
    maxWidth: "500px",
    textAlign: "center",
  },
  heading: {
    marginBottom: "2rem",
    color: "#0a2540",
  },
  input: {
    width: "100%",
    padding: "0.75rem 1rem",
    margin: "0.5rem 0",
    borderRadius: "0.5rem",
    border: "1px solid #ccc",
    fontSize: "1rem",
  },
  button: {
    width: "100%",
    padding: "0.75rem",
    marginTop: "0.5rem",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: "0.5rem",
    fontSize: "1rem",
    cursor: "pointer",
    transition: "background-color 0.3s ease",
  },
  answer: {
    background: "#eef1f4",
    padding: "1rem",
    borderRadius: "0.5rem",
    color: "#111",
    marginTop: "1rem",
  },
};

export default App;
