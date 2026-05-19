# Deployment Instructions: Private Decoupled Architecture

This guide explains how to deploy the Vectra SDK with a decoupled architecture: **Backend on a Private Hugging Face Space** and **Frontend on Cloudflare Pages**, using environment variables to keep your configuration secure.

## Phase 1: Deploying the Backend (Hugging Face Spaces)

1. **Create a New Space:**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - **SDK:** Select **Docker**.
   - **Visibility:** **Private**.

2. **Upload Files:**
   - Upload everything from `deployment/Backend/`.
   - **Wait for build.** Once "Running", copy the **Direct URL** from the "Embed this Space" or "Settings" menu (e.g., `https://username-space.hf.space`).

3. **Generate an Access Token:**
   - Go to your [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens).
   - Create a new **Read** token for this Space. **Copy it.**

## Phase 2: Secure Middleware (Cloudflare Workers) - RECOMMENDED

To keep your `HF_TOKEN` 100% hidden from the user's browser, you should use a **Cloudflare Worker** as a proxy.

1. **Create a Worker:**
   - Go to Cloudflare Dashboard -> **Workers & Pages** -> **Create application** -> **Create Worker**.
   - Name it `vectra-proxy` and click **Deploy**.

2. **Configure the Worker Code:**
   - Click **Edit Code**, paste the contents of `deployment/middleware_proxy.js`, and click **Save and Deploy**.

3. **Set Worker Secrets:**
   - Go to **Settings -> Variables**. Add three variables:
     - `API_BASE_URL`: Your Hugging Face Direct URL.
     - `HF_TOKEN`: Your Hugging Face Access Token (**Secret**).
     - `ALLOWED_ORIGIN`: Your Cloudflare Pages URL (e.g., `https://vectra-sdk.pages.dev`).

---

## Phase 3: Deploying the Frontend (Cloudflare Pages)

1. **Update API URL:**
   - In Cloudflare Pages **Settings -> Environment Variables**, set `API_BASE_URL` to your **Worker's URL** (e.g., `https://vectra-proxy.username.workers.dev`).
   - Leave `HF_TOKEN` empty (since the Worker handles it).

2. **Build Configuration:**
   - Build command: `sed -i "s|__API_BASE_URL__|$API_BASE_URL|g" index.html && sed -i "s|__HF_TOKEN__|$HF_TOKEN|g" index.html`

---

## Important Security Notes

- **Middleware Security**: By using the Cloudflare Worker, your `HF_TOKEN` stays on Cloudflare's servers and is **never sent to the user's browser**. This is the professional way to handle private API keys.
- **CORS**: The middleware script includes CORS headers, so your frontend can talk to it without issues.
