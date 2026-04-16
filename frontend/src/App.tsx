import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Asset = {
  id: string;
  original_filename: string;
  media_type: "pdf" | "audio" | "video";
  mime_type: string;
  summary: string;
  processing_status: string;
  error_message: string | null;
  duration_seconds: number | null;
  created_at: string;
  file_url: string;
  duration_label: string | null;
  text_preview: string;
};

type SourceSnippet = {
  asset_id: string;
  filename: string;
  media_type: string;
  excerpt: string;
  score: number;
  start_seconds: number | null;
  end_seconds: number | null;
  timestamp_label: string | null;
};

type TopicMatch = {
  asset_id: string;
  filename: string;
  media_type: string;
  label: string;
  excerpt: string;
  start_seconds: number;
  end_seconds: number | null;
  start_label: string;
  end_label: string | null;
};

type ChatResult = {
  answer: string;
  sources: SourceSnippet[];
  timestamp_matches: TopicMatch[];
};

type User = {
  id: string;
  username: string;
  created_at: string;
};

type AuthResponse = {
  access_token: string;
  token_type: string;
  user: User;
  api_key?: string | null;
};

type ApiKeyResponse = {
  api_key: string;
};

type StreamEvent =
  | { type: "answer_delta"; delta: string }
  | { type: "metadata"; sources: SourceSnippet[]; timestamp_matches: TopicMatch[] }
  | { type: "done" };

type AuthMode = "login" | "register";

const configuredApiBase = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL = configuredApiBase === undefined ? "http://localhost:8000" : configuredApiBase.replace(/\/$/, "");
const TOKEN_STORAGE_KEY = "mediamind-access-token";

function App() {
  const [token, setToken] = useState(() => window.localStorage.getItem(TOKEN_STORAGE_KEY) ?? "");
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authMode, setAuthMode] = useState<AuthMode>("register");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(Boolean(token));
  const [latestApiKey, setLatestApiKey] = useState<string | null>(null);

  const [assets, setAssets] = useState<Asset[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [pendingFiles, setPendingFiles] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [chatResult, setChatResult] = useState<ChatResult | null>(null);
  const [isAsking, setIsAsking] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
  const [topicQueries, setTopicQueries] = useState<Record<string, string>>({});
  const [topicResults, setTopicResults] = useState<Record<string, TopicMatch[]>>({});
  const [topicLoading, setTopicLoading] = useState<Record<string, boolean>>({});
  const [recentUploadIds, setRecentUploadIds] = useState<string[]>([]);
  const [showApiKey, setShowApiKey] = useState(false);
  const mediaRefs = useRef<Record<string, HTMLMediaElement | null>>({});
  const libraryRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!token) {
      setCurrentUser(null);
      setAuthLoading(false);
      return;
    }

    void hydrateSession(token);
  }, [token]);

  useEffect(() => {
    if (!currentUser) {
      setAssets([]);
      setSelectedAssetIds([]);
      setIsLoading(false);
      return;
    }

    void loadAssets();
  }, [currentUser]);

  const selectedCount = selectedAssetIds.length;
  const mediaCount = useMemo(
    () => assets.filter((asset) => asset.media_type === "audio" || asset.media_type === "video").length,
    [assets],
  );
  const hasUploadedAssets = assets.length > 0;
  const hasPendingFiles = pendingFiles.length > 0;
  const hasPendingMediaFiles = useMemo(
    () => pendingFiles.some((name) => /\.(mp3|wav|m4a|mp4|mov|avi|flac|ogg|webm|aac)$/i.test(name)),
    [pendingFiles],
  );
  const uploadButtonLabel =
    pendingFiles.length === 0
      ? "Upload and analyze"
      : pendingFiles.length === 1
        ? `Upload ${pendingFiles[0]}`
        : `Upload ${pendingFiles.length} files`;

  async function hydrateSession(activeToken: string) {
    setAuthLoading(true);
    try {
      const user = await requestJson<User>("/api/auth/me", { tokenOverride: activeToken });
      setCurrentUser(user);
      setAuthError(null);
    } catch (error) {
      clearSession(error instanceof Error ? error.message : null);
    } finally {
      setAuthLoading(false);
    }
  }

  async function requestJson<T>(
    path: string,
    options: {
      method?: string;
      body?: BodyInit | null;
      headers?: HeadersInit;
      tokenOverride?: string;
    } = {},
  ): Promise<T> {
    const headers = new Headers(options.headers);
    const activeToken = options.tokenOverride ?? token;
    if (activeToken) {
      headers.set("Authorization", `Bearer ${activeToken}`);
    }

    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method ?? "GET",
      body: options.body ?? null,
      headers,
    });

    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      if (response.status === 401) {
        clearSession("Your session expired. Sign in again.");
      }
      const detail =
        payload && typeof payload === "object" && "detail" in payload ? String(payload.detail) : "Request failed.";
      throw new Error(detail);
    }

    return payload as T;
  }

  async function loadAssets() {
    setIsLoading(true);
    setUploadError(null);
    try {
      const data = await requestJson<Asset[]>("/api/assets");
      setAssets(data);
      setSelectedAssetIds((current) => current.filter((assetId) => data.some((asset) => asset.id === assetId)));
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Failed to load assets.");
    } finally {
      setIsLoading(false);
    }
  }

  function persistSession(payload: AuthResponse) {
    window.localStorage.setItem(TOKEN_STORAGE_KEY, payload.access_token);
    setToken(payload.access_token);
    setCurrentUser(payload.user);
    setLatestApiKey(payload.api_key ?? null);
    setShowApiKey(false);
    setAssets([]);
    setSelectedAssetIds([]);
    setChatResult(null);
    setTopicResults({});
    setTopicQueries({});
    setUploadMessage(null);
    setUploadError(null);
    setPendingFiles([]);
    setRecentUploadIds([]);
  }

  function clearSession(message: string | null = null) {
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    setToken("");
    setCurrentUser(null);
    setLatestApiKey(null);
    setShowApiKey(false);
    setAssets([]);
    setSelectedAssetIds([]);
    setChatResult(null);
    setChatError(null);
    setTopicResults({});
    setTopicQueries({});
    setPendingFiles([]);
    setRecentUploadIds([]);
    setUploadMessage(null);
    setUploadError(null);
    setAuthError(message);
  }

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthLoading(true);
    setAuthError(null);

    try {
      const payload = await requestJson<AuthResponse>(`/api/auth/${authMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
        tokenOverride: "",
      });
      persistSession(payload);
      setPassword("");
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Authentication failed.");
    } finally {
      setAuthLoading(false);
    }
  }

  async function handleRotateApiKey() {
    setAuthError(null);
    try {
      const payload = await requestJson<ApiKeyResponse>("/api/auth/api-key", { method: "POST" });
      setLatestApiKey(payload.api_key);
      setShowApiKey(true);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Failed to rotate API key.");
    }
  }

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? [])
      .filter((file) => file.size > 0)
      .map((file) => file.name);
    setPendingFiles(selected);
    setUploadError(null);
    if (selected.length > 0) {
      setUploadMessage('Selected files are local only until you click "Upload and analyze."');
    } else {
      setUploadMessage(null);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const files = form.getAll("files").filter((entry): entry is File => entry instanceof File && entry.size > 0);
    if (files.length === 0) {
      setUploadError("Choose at least one PDF, audio, or video file.");
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadMessage(null);

    try {
      const uploaded: Asset[] = [];
      for (const file of files) {
        const payload = new FormData();
        payload.append("file", file);

        const headers = new Headers();
        if (token) {
          headers.set("Authorization", `Bearer ${token}`);
        }

        const response = await fetch(`${API_BASE_URL}/api/assets/upload`, {
          method: "POST",
          body: payload,
          headers,
        });

        const body = await response.json().catch(() => null);
        if (!response.ok) {
          if (response.status === 401) {
            clearSession("Your session expired. Sign in again.");
          }
          const detail = body && typeof body === "object" && "detail" in body ? String(body.detail) : "Upload failed.";
          throw new Error(detail);
        }

        const data = body as { asset: Asset };
        uploaded.push(data.asset);
      }

      setAssets((current) => {
        const merged = new Map<string, Asset>();
        for (const asset of uploaded) {
          merged.set(asset.id, asset);
        }
        for (const asset of current) {
          if (!merged.has(asset.id)) {
            merged.set(asset.id, asset);
          }
        }
        return [...merged.values()];
      });
      setSelectedAssetIds((current) => [...new Set([...uploaded.map((asset) => asset.id), ...current])]);
      setRecentUploadIds(uploaded.map((asset) => asset.id));
      setChatResult(null);
      setPendingFiles([]);
      setUploadMessage(
        `Uploaded ${uploaded.length} file${uploaded.length === 1 ? "" : "s"}. The generated summaries are now shown in the Content Library below.`,
      );
      formElement.reset();
      window.setTimeout(() => {
        libraryRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 50);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleQuestionSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    if (!hasUploadedAssets) {
      setChatError(
        hasPendingFiles
          ? 'Click "Upload and analyze" first. Picking a file only selects it locally.'
          : "Upload at least one file before asking a question.",
      );
      return;
    }

    setIsAsking(true);
    setChatError(null);
    setChatResult({ answer: "", sources: [], timestamp_matches: [] });

    try {
      const headers = new Headers({ "Content-Type": "application/json" });
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      }

      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          question,
          asset_ids: selectedAssetIds.length > 0 ? selectedAssetIds : null,
        }),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({ detail: "Question failed." }));
        if (response.status === 401) {
          clearSession("Your session expired. Sign in again.");
        }
        throw new Error(errorBody.detail ?? "Question failed.");
      }

      if (!response.body) {
        const fallback = await requestJson<ChatResult>("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question,
            asset_ids: selectedAssetIds.length > 0 ? selectedAssetIds : null,
          }),
        });
        setChatResult(fallback);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) {
            continue;
          }

          const eventPayload = JSON.parse(line) as StreamEvent;
          if (eventPayload.type === "answer_delta") {
            answer += eventPayload.delta;
            setChatResult((current) => ({
              answer,
              sources: current?.sources ?? [],
              timestamp_matches: current?.timestamp_matches ?? [],
            }));
          }

          if (eventPayload.type === "metadata") {
            setChatResult((current) => ({
              answer: current?.answer ?? answer,
              sources: eventPayload.sources,
              timestamp_matches: eventPayload.timestamp_matches,
            }));
          }
        }

        if (done) {
          if (buffer.trim()) {
            const eventPayload = JSON.parse(buffer) as StreamEvent;
            if (eventPayload.type === "metadata") {
              setChatResult((current) => ({
                answer: current?.answer ?? answer,
                sources: eventPayload.sources,
                timestamp_matches: eventPayload.timestamp_matches,
              }));
            }
          }
          break;
        }
      }
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Question failed.");
    } finally {
      setIsAsking(false);
    }
  }

  async function handleTopicSearch(assetId: string) {
    const topic = topicQueries[assetId]?.trim();
    if (!topic) {
      return;
    }

    setTopicLoading((current) => ({ ...current, [assetId]: true }));
    setUploadError(null);
    try {
      const data = await requestJson<TopicMatch[]>(`/api/assets/${assetId}/topics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic }),
      });
      setTopicResults((current) => ({ ...current, [assetId]: data }));
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "Topic search failed.");
    } finally {
      setTopicLoading((current) => ({ ...current, [assetId]: false }));
    }
  }

  function toggleSelection(assetId: string) {
    setSelectedAssetIds((current) =>
      current.includes(assetId) ? current.filter((value) => value !== assetId) : [...current, assetId],
    );
  }

  async function playMatch(match: TopicMatch | SourceSnippet) {
    const player = mediaRefs.current[match.asset_id];
    if (!player || match.start_seconds === null) {
      return;
    }
    player.currentTime = match.start_seconds;
    await player.play().catch(() => undefined);
  }

  const authPanel = (
    <section className="panel auth-panel">
      <div className="panel-header">
        <h2>{authMode === "register" ? "Create your workspace" : "Sign back in"}</h2>
        <p>Every account has its own uploads, JWT session, and optional API key for direct API access.</p>
      </div>
      <form onSubmit={handleAuthSubmit} className="stack">
        <div className="mode-switch">
          <button
            type="button"
            className={authMode === "register" ? "secondary active-pill" : "secondary"}
            onClick={() => setAuthMode("register")}
          >
            Register
          </button>
          <button
            type="button"
            className={authMode === "login" ? "secondary active-pill" : "secondary"}
            onClick={() => setAuthMode("login")}
          >
            Login
          </button>
        </div>
        <input
          type="text"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          placeholder="username"
          autoComplete="username"
          minLength={3}
          maxLength={32}
        />
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="password"
          autoComplete={authMode === "register" ? "new-password" : "current-password"}
          minLength={8}
        />
        <button type="submit" disabled={authLoading}>
          {authLoading ? "Working..." : authMode === "register" ? "Create account" : "Sign in"}
        </button>
      </form>
      {authError ? <p className="error">{authError}</p> : null}
      {latestApiKey ? (
        <div className="developer-tools">
          <button type="button" className="secondary" onClick={() => setShowApiKey((current) => !current)}>
            {showApiKey ? "Hide developer API key" : "Show developer API key"}
          </button>
          {showApiKey ? (
            <div className="api-key-card">
              <strong>Latest API key</strong>
              <code>{latestApiKey}</code>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );

  return (
    <div className="shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">SDE-1 Assignment Build</p>
          <h1>MediaMind QA</h1>
          <p className="lead">
            Upload PDFs, audio, and video, then ask grounded questions, review summaries, and jump into the exact
            timestamp that supports an answer.
          </p>
          {currentUser ? (
            <div className="account-row">
              <span className="account-pill">Signed in as {currentUser.username}</span>
              <button type="button" className="secondary" onClick={() => void handleRotateApiKey()}>
                Rotate API key
              </button>
              {latestApiKey ? (
                <button type="button" className="secondary" onClick={() => setShowApiKey((current) => !current)}>
                  {showApiKey ? "Hide API key" : "Show API key"}
                </button>
              ) : null}
              <button type="button" className="secondary" onClick={() => clearSession()}>
                Sign out
              </button>
            </div>
          ) : authLoading ? (
            <div className="account-row">
              <span className="account-pill">Checking saved session...</span>
            </div>
          ) : null}
        </div>
        <div className="hero-panel">
          <div>
            <span className="metric">{assets.length}</span>
            <span className="label">Uploaded files</span>
          </div>
          <div>
            <span className="metric">{mediaCount}</span>
            <span className="label">Media assets</span>
          </div>
          <div>
            <span className="metric">{selectedCount}</span>
            <span className="label">Scoped for chat</span>
          </div>
        </div>
      </header>

      {!currentUser ? (
        <main className="layout auth-layout">
          {authPanel}
          <section className="panel chat-panel">
            <div className="panel-header">
              <h2>What’s Included</h2>
              <p>This build now covers the missing backend and infrastructure work, not just the UI flow.</p>
            </div>
            <ul className="feature-list">
              <li>JWT-based multi-user authentication plus rotatable API keys.</li>
              <li>Vector-backed retrieval with FAISS support and safe in-memory fallback.</li>
              <li>Streaming chat responses over a newline-delimited event stream.</li>
              <li>Redis-ready caching and rate limiting with local in-memory fallback.</li>
            </ul>
          </section>
        </main>
      ) : (
        <main className="layout">
          <section className="panel upload-panel">
            <div className="panel-header">
              <h2>Ingest Content</h2>
              <p>Process files immediately, cache repeated queries, and index content for semantic retrieval.</p>
            </div>
            <form onSubmit={handleUpload} className="stack">
              <label className="dropzone">
                <span>Choose PDF, audio, or video files</span>
                <input
                  name="files"
                  type="file"
                  accept=".pdf,audio/*,video/*"
                  multiple
                  onChange={handleFileSelection}
                />
              </label>
              {hasPendingFiles ? (
                <div className="selection-card">
                  <strong>Selected files</strong>
                  <ul className="selection-list">
                    {pendingFiles.map((file) => (
                      <li key={file}>{file}</li>
                    ))}
                  </ul>
                  <p className="hint-text">These files are not uploaded yet. Click the button below to process them.</p>
                </div>
              ) : (
                <p className="hint-text">Choose a file, then click "Upload and analyze" to generate summaries.</p>
              )}
              <button type="submit" disabled={isUploading || !hasPendingFiles}>
                {isUploading ? "Processing..." : uploadButtonLabel}
              </button>
            </form>
            {uploadMessage ? <p className="info-text">{uploadMessage}</p> : null}
            {hasPendingMediaFiles ? (
              <p className="hint-text">
                Audio/video uploads require an OpenAI API key to transcribe and index the media for search, topics, and clip playback. If OPENAI_API_KEY is not configured, media upload may fail or be incomplete.
              </p>
            ) : null}
            {uploadError ? <p className="error">{uploadError}</p> : null}
            {showApiKey && latestApiKey ? (
              <div className="api-key-card">
                <strong>Developer API key</strong>
                <code>{latestApiKey}</code>
              </div>
            ) : null}
          </section>

          <section className="panel chat-panel">
            <div className="panel-header">
              <h2>Ask the Assistant</h2>
              <p>Responses stream in real time and stay grounded to the selected assets.</p>
            </div>
            <form onSubmit={handleQuestionSubmit} className="stack">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={4}
                placeholder={
                  hasUploadedAssets
                    ? "What is discussed about quarterly revenue in the uploaded files?"
                    : "Upload a file first, then ask a grounded question about it."
                }
              />
              <button type="submit" disabled={isAsking || !hasUploadedAssets}>
                {isAsking ? "Streaming..." : hasUploadedAssets ? "Ask question" : "Upload a file first"}
              </button>
            </form>
            {!hasUploadedAssets ? (
              <p className="hint-text">
                {hasPendingFiles
                  ? "You selected a file, but it is not part of the searchable library yet."
                  : "The assistant only answers from files that have already been uploaded and indexed."}
              </p>
            ) : null}
            {chatError ? <p className="error">{chatError}</p> : null}
            {chatResult ? (
              <div className="chat-result">
                <div className="answer-card">
                  <h3>Answer</h3>
                  <p>{chatResult.answer || "Waiting for answer..."}</p>
                </div>
                <div className="match-grid">
                  <div>
                    <h3>Sources</h3>
                    <ul className="chip-list">
                      {chatResult.sources.length === 0 ? (
                        <li className="empty-state">Sources will appear as soon as the answer completes.</li>
                      ) : (
                        chatResult.sources.map((source, index) => (
                          <li key={`${source.asset_id}-${index}`} className="match-card">
                            <div className="match-meta">
                              <strong>{source.filename}</strong>
                              <span>score {source.score.toFixed(2)}</span>
                            </div>
                            <p>{source.excerpt}</p>
                            {source.start_seconds !== null ? (
                              <button type="button" className="secondary" onClick={() => void playMatch(source)}>
                                Play from {source.timestamp_label}
                              </button>
                            ) : null}
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                  <div>
                    <h3>Timestamp Matches</h3>
                    <ul className="chip-list">
                      {chatResult.timestamp_matches.length === 0 ? (
                        <li className="empty-state">No timestamped media matched this answer.</li>
                      ) : (
                        chatResult.timestamp_matches.map((match, index) => (
                          <li key={`${match.asset_id}-${index}`} className="match-card">
                            <div className="match-meta">
                              <strong>{match.filename}</strong>
                              <span>{match.start_label}</span>
                            </div>
                            <p>{match.excerpt}</p>
                            <button type="button" className="secondary" onClick={() => void playMatch(match)}>
                              Play clip
                            </button>
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                </div>
              </div>
            ) : null}
          </section>

          <section className="panel library-panel" ref={libraryRef}>
            <div className="panel-header">
              <h2>Content Library</h2>
              <p>Review summaries, scope chat context, and search for timestamped topics inside media files.</p>
            </div>
            {uploadMessage && hasUploadedAssets ? <p className="info-text">{uploadMessage}</p> : null}
            {isLoading ? (
              <p className="empty-state">Loading uploaded assets...</p>
            ) : assets.length === 0 ? (
              <p className="empty-state">No content yet. Upload a PDF, audio, or video file to begin.</p>
            ) : (
              <div className="asset-grid">
                {assets.map((asset) => {
                  const isSelected = selectedAssetIds.includes(asset.id);
                  const topicMatches = topicResults[asset.id] ?? [];
                  const isMedia = asset.media_type === "audio" || asset.media_type === "video";

                  return (
                    <article
                      key={asset.id}
                      className={`asset-card ${isSelected ? "selected" : ""} ${recentUploadIds.includes(asset.id) ? "fresh" : ""}`}
                    >
                      <div className="asset-head">
                        <div>
                          <p className="asset-type">{asset.media_type.toUpperCase()}</p>
                          <h3>{asset.original_filename}</h3>
                        </div>
                        <label className="select-toggle">
                          <input type="checkbox" checked={isSelected} onChange={() => toggleSelection(asset.id)} />
                          Scope chat
                        </label>
                      </div>

                      <div className="status-row">
                        <span className={`status-pill ${asset.processing_status}`}>{asset.processing_status}</span>
                        {asset.duration_label ? <span className="meta-line">Duration: {asset.duration_label}</span> : null}
                      </div>

                      <p className="summary">{asset.summary}</p>
                      <p className="preview">{asset.text_preview}</p>
                      {asset.error_message ? <p className="error">{asset.error_message}</p> : null}

                      {asset.media_type === "video" ? (
                        <video
                          className="media-frame"
                          controls
                          src={asset.file_url}
                          ref={(node) => {
                            mediaRefs.current[asset.id] = node;
                          }}
                        />
                      ) : null}

                      {asset.media_type === "audio" ? (
                        <audio
                          className="media-frame"
                          controls
                          src={asset.file_url}
                          ref={(node) => {
                            mediaRefs.current[asset.id] = node;
                          }}
                        />
                      ) : null}

                      {asset.media_type === "pdf" ? (
                        <a className="secondary link-button" href={asset.file_url} target="_blank" rel="noreferrer">
                          Open PDF
                        </a>
                      ) : null}

                      {isMedia ? (
                        <div className="topic-box">
                          <div className="topic-row">
                            <input
                              type="text"
                              value={topicQueries[asset.id] ?? ""}
                              onChange={(event) =>
                                setTopicQueries((current) => ({ ...current, [asset.id]: event.target.value }))
                              }
                              placeholder="Find a topic and get timestamps"
                            />
                            <button
                              type="button"
                              className="secondary"
                              disabled={topicLoading[asset.id]}
                              onClick={() => void handleTopicSearch(asset.id)}
                            >
                              {topicLoading[asset.id] ? "Searching..." : "Find"}
                            </button>
                          </div>
                          <ul className="chip-list compact">
                            {topicMatches.length === 0 ? (
                              <li className="empty-state">Search within this recording to fetch clip timestamps.</li>
                            ) : (
                              topicMatches.map((match, index) => (
                                <li key={`${match.asset_id}-${index}`} className="match-card">
                                  <div className="match-meta">
                                    <strong>{match.start_label}</strong>
                                    <span>{match.end_label ?? "clip"}</span>
                                  </div>
                                  <p>{match.excerpt}</p>
                                  <button type="button" className="secondary" onClick={() => void playMatch(match)}>
                                    Play
                                  </button>
                                </li>
                              ))
                            )}
                          </ul>
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  );
}

export default App;
