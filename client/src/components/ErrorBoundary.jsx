import React from "react";

/**
 * React Error Boundary — catches render/lifecycle errors and shows a
 * fallback UI instead of a blank white screen.
 *
 * Usage:
 *   <ErrorBoundary>
 *     <App />
 *   </ErrorBoundary>
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, info: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    this.setState({ info });
    // In production you would send this to an error tracker (Sentry, etc.)
    console.error("[ErrorBoundary]", error, info?.componentStack);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    return (
      <div style={styles.container}>
        <div style={styles.card}>
          <h1 style={styles.heading}>Something went wrong</h1>
          <p style={styles.sub}>
            The application crashed unexpectedly. Please refresh the page.
          </p>
          <pre style={styles.detail}>
            {this.state.error?.message || String(this.state.error)}
          </pre>
          <button
            style={styles.btn}
            onClick={() => window.location.reload()}
          >
            Reload page
          </button>
        </div>
      </div>
    );
  }
}

const styles = {
  container: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#0f1117",
    padding: "2rem",
  },
  card: {
    background: "#1a1d27",
    border: "1px solid #ef4444",
    borderRadius: "12px",
    padding: "2.5rem",
    maxWidth: "560px",
    width: "100%",
    textAlign: "center",
  },
  heading: {
    color: "#ef4444",
    fontSize: "1.5rem",
    marginBottom: "0.75rem",
    fontWeight: 700,
  },
  sub: {
    color: "#94a3b8",
    marginBottom: "1.25rem",
    lineHeight: 1.6,
  },
  detail: {
    background: "#0f1117",
    color: "#f97316",
    borderRadius: "8px",
    padding: "1rem",
    fontSize: "0.8rem",
    textAlign: "left",
    overflowX: "auto",
    marginBottom: "1.5rem",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  btn: {
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    padding: "0.6rem 1.5rem",
    fontSize: "0.95rem",
    cursor: "pointer",
    fontWeight: 600,
  },
};

export default ErrorBoundary;
